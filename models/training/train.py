import os
import time
from copy import copy
from typing import Dict, List, Tuple, Any

import dgl
import numpy as np
import pandas
import torch
import torch.optim as opt
import wandb
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from tqdm import tqdm

from models.dataset.dataset_creation import create_dataloader
from models.training.checkpoint import save_checkpoint, load_checkpoint, save_csv, translate_zs_paper_model_state_dict
from models.training.extended_evaluation import slice_evaluation_output
from models.training.metrics import MAPE, RMSE, QError, ProcentualError
from models.training.pca_analysis import perform_pca_analysis
from models.training.utils import batch_to, find_early_stopping_metric
from models.zero_shot_models.specific_models.model import zero_shot_models


def train_epoch(epoch_stats, train_loader, model, optimizer, max_epoch_tuples, custom_batch_to=batch_to,
                pt_profiler=None, gradient_norm: bool = False, test_with_count_edges_msg_aggr: bool = False):
    model.train()

    # run remaining batches
    train_start_t = time.perf_counter()
    losses = []
    errs = []

    error_ctr = 0

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if max_epoch_tuples is not None and batch_idx * train_loader.batch_size > max_epoch_tuples:
            break

        input_model, label, stats, sample_idxs = custom_batch_to(batch, model.device, model.label_norm)

        # print(input_model[0])
        # print(input_model[1])
        #
        # for ntype in input_model[0].ntypes:
        #     print(f'{ntype}: {input_model[0].nodes[ntype].data}')
        #
        # for etype in input_model[0].etypes:
        #     print(f'{etype}: {input_model[0].edges[etype].data}')

        optimizer.zero_grad()

        out = model(input_model)

        # import sys
        # sys.exit(0)

        if model.return_graph_repr and model.return_udf_repr:
            output, graph_repr, udf_repr, feat_dict = out
        elif model.return_graph_repr:
            output, graph_repr, feat_dict = out
        elif model.return_udf_repr:
            output, udf_repr, feat_dict = out
        else:
            output, feat_dict = out

        if torch.isnan(output).any():
            raise ValueError('Output was NaN')
        loss = model.loss_fxn(output, label)
        if torch.isnan(loss):
            raise ValueError('Loss was NaN')
        if not test_with_count_edges_msg_aggr:
            # otherwise an error that gradients cannot be computed are thrown
            loss.backward()

        # apply gradient clipping
        if gradient_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        loss = loss.detach().cpu().numpy()
        output = output.detach().cpu().numpy().reshape(-1)
        label = label.detach().cpu().numpy().reshape(-1)
        errs = np.concatenate((errs, output - label))
        losses.append(loss)

        if pt_profiler is not None:
            pt_profiler.step()

        if test_with_count_edges_msg_aggr:
            # compare with count of edges which is output
            num_nodes_list = stats['graph_num_nodes']
            num_edges_list = stats['graph_num_edges']

            assert len(num_nodes_list) == len(num_edges_list), f'{len(num_nodes_list)} vs {len(num_edges_list)}'
            assert len(num_nodes_list) == len(output), f'{len(num_nodes_list)} vs {len(output)}'

            for i, (num_nodes, num_edges, out) in enumerate(zip(num_nodes_list, num_edges_list, output)):
                rel_dif = abs(num_nodes - out) / num_nodes
                if rel_dif >= 0.0001:
                    graph = input_model[0]
                    for ntype, feat in feat_dict.items():
                        graph.nodes[ntype].data['h'] = feat

                    graph = graph.to('cpu')

                    dgl.save_graphs(f'tmp/graph.bin', [graph], {'label': torch.tensor([out])})
                    # nx_graph = dgl.to_networkx(batch[0])
                    # # save graph
                    # with open(f'tmp/graph.pkl', 'wb') as f:
                    #     pickle.dump(nx_graph, f)
                    # assert rel_dif<0.0001, f'{batch_idx}: {num_nodes} != {out} (num edges: {num_edges})'
                    # print(f'{batch_idx}: {num_nodes} != {out} (num edges: {num_edges})',flush=True)
                    error_ctr += 1

                    if error_ctr % 10 == 0 and error_ctr > 0:
                        print(f'Error count: {error_ctr}', flush=True)

    if test_with_count_edges_msg_aggr:
        print(f'Error count: {error_ctr}', flush=True)

    mean_loss = np.mean(losses)
    mean_rmse = np.sqrt(np.mean(np.square(errs)))
    # print(f"Train Loss: {mean_loss:.2f}")
    # print(f"Train RMSE: {mean_rmse:.2f}")
    epoch_stats.update(train_time=time.perf_counter() - train_start_t, mean_loss=mean_loss, mean_rmse=mean_rmse)


def run_inference(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, max_epoch_tuples: int,
                  custom_batch_to=batch_to):
    model.eval()
    with torch.autograd.no_grad():
        val_loss = torch.Tensor([0])
        labels = []
        preds = []
        graph_reprs = []
        udf_reprs = []
        sample_idxs = []

        # stats
        database_name = []
        num_joins = []
        num_filters = []
        udf_num_np_calls = []
        udf_num_math_calls = []
        udf_num_comp_nodes = []
        udf_num_branches = []
        udf_num_loops = []
        sql_list = []
        udf_pos_in_query = []
        udf_in_card = []
        udf_filter_num_logicals = []
        udf_filter_num_literals = []

        # evaluate test set using model
        test_start_t = time.perf_counter()
        val_num_tuples = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if max_epoch_tuples is not None and batch_idx * data_loader.batch_size > max_epoch_tuples:
                break

            val_num_tuples += data_loader.batch_size

            input_model, label, stats, sample_idxs_batch = custom_batch_to(batch, model.device, model.label_norm)
            sample_idxs += sample_idxs_batch

            out = model(input_model)

            graph_repr = None
            udf_repr = None
            if model.return_graph_repr and model.return_udf_repr:
                output, graph_repr, udf_repr, feat_dict = out
                assert len(graph_repr) == len(label), f'{len(graph_repr)} vs {len(label)}'
                assert len(udf_repr) == len(label), f'{len(udf_repr)} vs {len(label)}'
            elif model.return_graph_repr:
                output, graph_repr, feat_dict = out
                assert len(graph_repr) == len(label), f'{len(graph_repr)} vs {len(label)}'
            elif model.return_udf_repr:
                output, udf_repr, feat_dict = out
                assert len(udf_repr) == len(label), f'{len(udf_repr)} vs {len(label)}'
            else:
                output, feat_dict = out

            if graph_repr is not None:
                graph_reprs.append(graph_repr.cpu().numpy())
            if udf_repr is not None:
                udf_reprs.append(udf_repr.cpu().numpy())

            # sum up mean batch losses
            val_loss += model.loss_fxn(output, label).cpu()

            # inverse transform the predictions and labels
            curr_pred = output.cpu().numpy()
            curr_label = label.cpu().numpy()
            if model.label_norm is not None:
                curr_pred = model.label_norm.inverse_transform(curr_pred)
                curr_label = model.label_norm.inverse_transform(curr_label.reshape(-1, 1))
                curr_label = curr_label.reshape(-1)

            preds.append(curr_pred.reshape(-1))
            labels.append(curr_label.reshape(-1))

            # stats
            num_joins.append(stats['num_joins'])
            num_filters.append(stats['num_filters'])
            udf_num_np_calls.append(stats['udf_num_np_calls'])
            udf_num_math_calls.append(stats['udf_num_math_calls'])
            udf_num_comp_nodes.append(stats['udf_num_comp_nodes'])
            udf_num_branches.append(stats['udf_num_branches'])
            udf_num_loops.append(stats['udf_num_loops'])
            sql_list.append(stats['sql_list'])
            udf_pos_in_query.append(stats['udf_pos_in_query'])
            udf_in_card.append(stats['udf_in_card'])
            database_name.append(stats['database_name'])
            udf_filter_num_logicals.append(stats['udf_filter_num_logicals'])
            udf_filter_num_literals.append(stats['udf_filter_num_literals'])

    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)

    if model.return_graph_repr:
        graph_reprs = np.concatenate(graph_reprs, axis=0)
        assert graph_reprs.shape[0] == labels.shape[0], f'{graph_reprs.shape} vs {labels.shape}'
    if model.return_udf_repr:
        udf_reprs = np.concatenate(udf_reprs, axis=0)
        assert udf_reprs.shape[0] == labels.shape[0], f'{udf_reprs.shape} vs {labels.shape}'

    # stats
    num_joins = np.concatenate(num_joins, axis=0)
    num_filters = np.concatenate(num_filters, axis=0)
    udf_num_np_calls = np.concatenate(udf_num_np_calls, axis=0)
    udf_num_math_calls = np.concatenate(udf_num_math_calls, axis=0)
    udf_num_comp_nodes = np.concatenate(udf_num_comp_nodes, axis=0)
    udf_num_branches = np.concatenate(udf_num_branches, axis=0)
    udf_num_loops = np.concatenate(udf_num_loops, axis=0)
    sql_list = np.concatenate(sql_list, axis=0)
    udf_pos_in_query = np.concatenate(udf_pos_in_query, axis=0)
    udf_in_card = np.concatenate(udf_in_card, axis=0)
    database_name = np.concatenate(database_name, axis=0)
    udf_filter_num_logicals = np.concatenate(udf_filter_num_logicals, axis=0)
    udf_filter_num_literals = np.concatenate(udf_filter_num_literals, axis=0)

    stats = {
        'sql': sql_list,
        'num_joins': num_joins,
        'num_filters': num_filters,
        'udf_num_np_calls': udf_num_np_calls,
        'udf_num_math_calls': udf_num_math_calls,
        'udf_num_comp_nodes': udf_num_comp_nodes,
        'udf_num_branches': udf_num_branches,
        'udf_num_loops': udf_num_loops,
        'udf_pos_in_query': udf_pos_in_query,
        'udf_in_card': udf_in_card,
        'labels': labels,
        'preds': preds,
        'database_name': database_name,
        'udf_filter_num_logicals': udf_filter_num_logicals,
        'udf_filter_num_literals': udf_filter_num_literals
    }

    return labels, preds, graph_reprs, udf_reprs, sample_idxs, val_num_tuples, test_start_t, val_loss, stats


def validate_model(data_loader, model, validate_stats: Dict, epoch=0, metrics=None, max_epoch_tuples=None,
                   verbose=False, log_all_queries=False, extended_evaluation: bool = False,
                   prefix: str = None, is_test_loader: bool = False, log_to_wandb: bool = True) -> \
        Tuple[bool, Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    labels, preds, graph_reprs, udf_reprs, sample_idxs, val_num_tuples, test_start_t, val_loss, stats = run_inference(
        data_loader, model,
        max_epoch_tuples)

    validate_stats[f'{prefix}_time'] = time.perf_counter() - test_start_t
    validate_stats[f'{prefix}_num_tuples'] = val_num_tuples
    val_loss = (val_loss.cpu() / min(val_num_tuples, len(data_loader))).item()
    print(f'{prefix}_loss epoch {epoch}: {val_loss}')
    validate_stats[f'{prefix}_loss'] = val_loss

    if verbose:
        print(f'labels: {labels}')
        print(f'preds: {preds}')

    validate_stats[f'{prefix}_std'] = np.std(labels)
    if log_all_queries:
        # create a pandas dataframe with all queries
        data_dict = {
            'label': labels,
            'pred': preds,
            'sample_idxs': sample_idxs,
        }
        for key, value in stats.items():
            data_dict[key] = value

        table = pandas.DataFrame(data_dict)
        table = wandb.Table(dataframe=table)

        validate_stats[f'{prefix}_data'] = table

    # save best model for every metric
    any_best_metric = False
    if metrics is not None:
        for metric in metrics:
            metric_stats = dict()
            best_seen = metric.evaluate(metrics_dict=metric_stats, model=model, labels=labels, preds=preds,
                                        probs=None, update_best_seen=not is_test_loader)
            if best_seen and metric.early_stopping_metric:
                any_best_metric = True
                print(f"New best model for {metric.metric_name}")

            # copy metric stats over
            for key, value in metric_stats.items():
                # replace prefix
                assert key.startswith(metric.metric_prefix), f'{key} vs {metric.metric_prefix}'
                new_key = f'{prefix}_{key[len(metric.metric_prefix):]}'
                validate_stats[new_key] = value

    wandb_plots = dict()
    if log_to_wandb:
        # wandb_table = wandb.Table(data=[[label, pred] for label, pred in zip(labels, preds)],
        #                           columns=["label", "pred"])
        # labels_hist = wandb.plot.histogram(wandb_table, "label", title=f"{prefix} label histogram")
        #
        # preds_hist = wandb.plot.histogram(wandb_table, "pred", title=f"{prefix} pred histogram")

        # wandb_plots = {f'{prefix} labels hist': labels_hist, f'{prefix} preds hist': preds_hist}

        # compute a histogram of the predictions and labels
        data_array = np.array([preds, labels]).T.astype(np.float32)
        plt.close()
        plt.hist(data_array, label=["predictions", "labels"])
        plt.xlabel("runtime (s)")
        plt.ylabel("count")
        plt.legend(loc="upper right")

        wandb_plots[f'{prefix} predictions vs. labels'] = wandb.Image(plt)

    if extended_evaluation:
        validate_stats.update(
            slice_evaluation_output(labels=labels, predictions=preds, num_joins=stats['num_joins'],
                                    num_filters=stats['num_filters'],
                                    udf_num_np_calls=stats['udf_num_np_calls'],
                                    udf_num_math_calls=stats['udf_num_math_calls'],
                                    udf_num_comp_nodes=stats['udf_num_comp_nodes'],
                                    udf_num_branches=stats['udf_num_branches'],
                                    udf_num_loops=stats['udf_num_loops'], udf_pos_in_query=stats['udf_pos_in_query'],
                                    udf_in_card=stats['udf_in_card'],
                                    udf_filter_num_literals=stats['udf_filter_num_literals'],
                                    udf_filter_num_logicals=stats['udf_filter_num_logicals'],
                                    prefix=prefix, log_to_wandb=log_to_wandb))

    # analyze worst entry
    q_errors = np.maximum(labels / preds, preds / labels)
    worst_idx = np.argmax(q_errors)
    print(
        f"Worst entry: label={labels[worst_idx]} vs. pred={preds[worst_idx]} (qError = {q_errors[worst_idx]})\n{stats['database_name'][worst_idx]}: {stats['sql'][worst_idx]}")

    return any_best_metric, wandb_plots, graph_reprs, udf_reprs, labels, preds, stats


class StopEarly(Exception):
    pass


def extract_test_paths(path: str, card: str, target_dir: str, filename_model: str):
    test_workload = os.path.basename(path).replace('.json', '')

    # check if a test workload with the same name already exists
    # rename the workload
    if 'pullup' in path:
        test_workload = test_workload + '_pullup'
    elif 'pushdown' in path:
        test_workload = test_workload + '_pushdown'
    elif ('noudf' in path or '_no_udf' in path) and '_large' not in path:
        test_workload = test_workload + '_no_udf'
    elif ('noudf' in path or '_no_udf' in path) and '_large' in path:
        test_workload = test_workload + '_no_udf_large'
    elif 'v101' in path:
        test_workload = test_workload + '_v101'

    test_workload = test_workload + f'_{card}'

    test_path = os.path.join(target_dir, f'test_{filename_model}_{test_workload}.csv')

    return test_workload, test_path


def train_model(workload_runs,
                test_workload_runs,
                statistics_file,
                target_dir,
                filename_model,
                featurization: Any,
                batch_size: int,
                card_type: str,
                pretrained_model_artifact_dir: str = None,
                pretrained_model_filename: str = None,
                optimizer_class_name='Adam',
                optimizer_kwargs=None,
                final_mlp_kwargs=None,
                node_type_kwargs=None,
                model_kwargs=None,
                tree_layer_kwargs=None,
                output_dim=1,
                epochs=0,
                ft_epochs_udf_only=0,
                ft_epochs=0,
                device='cpu',
                max_epoch_tuples=100000,
                num_workers=1,
                early_stopping_patience=20,
                database=None,
                limit_queries=None,
                limit_queries_affected_wl=None,
                skip_train=False,
                seed=0, stratification_prioritize_loops: bool = False,
                register_at_wandb: bool = False, additional_wandb_stats: Dict = None, pt_profile: bool = False,
                offset_np_import: int = 0, mp_ignore_udf: bool = False, apply_gradient_norm: bool = False,
                stratify_dataset_by_runtimes: bool = False, stratify_per_database_by_runtimes: bool = False,
                max_runtime: int = None, apply_pca_evaluation: bool = False,
                test_only: bool = False, multi_label_keep_duplicates: bool = False, zs_paper_dataset: bool = False,
                plans_have_no_udf: bool = False, apply_lr_reduction_on_plateau: bool = False,
                train_udf_graph_against_udf_runtime: bool = False, work_with_udf_repr: bool = False,
                valtest: bool = True, w_loop_end_node: bool = False,
                add_loop_loopend_edge: bool = False, card_est_assume_lazy_eval: bool = True,
                include_no_udf_data: bool = False, test_with_count_edges_msg_aggr: bool = False,
                min_runtime_ms: int = 100, skip_udf: bool = False, filter_plans: Dict[str, int] = None):
    if model_kwargs is None:
        model_kwargs = dict()

    # seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert card_type in ['est', 'act', 'dd']

    # create a dataset
    loss_class_name = final_mlp_kwargs['loss_class_name']
    label_norm, feature_statistics, train_loader, val_loader, train_loader_udf_only, val_loader_udf_only, test_loaders, test_loader_names, finetune_loaders = \
        create_dataloader(workload_runs, test_workload_runs, statistics_file, featurization, database,
                          val_ratio=0.15, finetune_ratio=0.0, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers,
                          pin_memory=False, limit_queries=limit_queries,
                          limit_queries_affected_wl=limit_queries_affected_wl, loss_class_name=loss_class_name,
                          offset_np_import=offset_np_import, stratify_dataset_by_runtimes=stratify_dataset_by_runtimes,
                          stratify_per_database_by_runtimes=stratify_per_database_by_runtimes,
                          max_runtime=max_runtime, multi_label_keep_duplicates=multi_label_keep_duplicates,
                          zs_paper_dataset=zs_paper_dataset,
                          train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
                          w_loop_end_node=w_loop_end_node, add_loop_loopend_edge=add_loop_loopend_edge,
                          card_est_assume_lazy_eval=card_est_assume_lazy_eval, min_runtime_ms=min_runtime_ms,
                          card_type_in_udf=card_type, card_type_above_udf=card_type, card_type_below_udf=card_type,
                          plans_have_no_udf=plans_have_no_udf,
                          stratification_prioritize_loops=stratification_prioritize_loops, skip_udf=skip_udf,
                          filter_plans=filter_plans)

    assert len(test_loaders) > 0, "No test loaders found"

    if loss_class_name == 'QLoss':
        metrics = [RMSE(), MAPE(), QError(percentile=50, early_stopping_metric=True), QError(percentile=95),
                   QError(percentile=100), ProcentualError()]
    elif loss_class_name == 'MSELoss':
        metrics = [RMSE(early_stopping_metric=True), MAPE(), QError(percentile=50), QError(percentile=95),
                   QError(percentile=100), ProcentualError()]
    elif loss_class_name == 'ProcentualLoss':
        metrics = [RMSE(), MAPE(), QError(percentile=50), QError(percentile=95),
                   QError(percentile=100), ProcentualError(early_stopping_metric=True)]
    else:
        raise ValueError(f'Unknown loss class {loss_class_name}')

    # create zero shot model dependent on database
    model = zero_shot_models[database](device=device, final_mlp_kwargs=final_mlp_kwargs,
                                       node_type_kwargs=node_type_kwargs, output_dim=output_dim,
                                       feature_statistics=feature_statistics,
                                       tree_layer_kwargs=tree_layer_kwargs,
                                       featurization=featurization,
                                       label_norm=label_norm, mp_ignore_udf=mp_ignore_udf,
                                       return_graph_repr=apply_pca_evaluation,
                                       return_udf_repr=apply_pca_evaluation and not plans_have_no_udf and not include_no_udf_data,
                                       plans_have_no_udf=plans_have_no_udf,
                                       train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
                                       work_with_udf_repr=work_with_udf_repr,
                                       test_with_count_edges_msg_aggr=test_with_count_edges_msg_aggr,
                                       **model_kwargs)
    # move to gpu
    model = model.to(model.device)
    # print(model)
    optimizer = opt.__dict__[optimizer_class_name](model.parameters(), **optimizer_kwargs)
    if apply_lr_reduction_on_plateau:
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', threshold=1e-2)
    else:
        lr_scheduler = None

    initial_lr = optimizer.param_groups[0]['lr']

    if pretrained_model_artifact_dir is not None and pretrained_model_filename is not None:
        csv_stats, epochs_wo_improvement, epoch, model, optimizer, lr_scheduler, metrics, finished = \
            load_checkpoint(model, pretrained_model_artifact_dir, pretrained_model_filename, optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            metrics=metrics, filetype='.pt', zs_paper_model=zs_paper_dataset)
    else:
        csv_stats, epochs_wo_improvement, epoch, model, optimizer, lr_scheduler, metrics, finished = \
            load_checkpoint(model, target_dir, filename_model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                            metrics=metrics, filetype='.pt', zs_paper_model=zs_paper_dataset)

    if test_only:
        assert epoch > 0
        finished = True

    if register_at_wandb and not finished and not skip_train:
        watched = True
        wandb.watch(model, log='all', log_freq=len(train_loader) * 5, log_graph=True)
    else:
        watched = False

    if pt_profile:
        log_path = f'{target_dir}/{wandb.run.name}'
        print(f'Profile with pytorch profiler to logdir {log_path}')
        # setup pytorch profiler with tensorboard
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
            with_stack=True, record_shapes=True)

        prof.start()
    else:
        prof = None

    # apply learning rate scheduler to pre-training only if there is no finetuning
    if ft_epochs_udf_only is not None and ft_epochs_udf_only > 0:
        pretrain_lr = None
        ft_lr = lr_scheduler
    else:
        pretrain_lr = lr_scheduler
        ft_lr = None

    # train an actual model
    while epoch < epochs and not finished and not skip_train:
        try:
            train_epoch_fn(epoch=epoch, train_loader=train_loader, val_loader=val_loader,
                           model=model, optimizer=optimizer, max_epoch_tuples=max_epoch_tuples, prof=prof,
                           apply_gradient_norm=apply_gradient_norm, metrics=metrics, lr_scheduler=pretrain_lr,
                           epochs_wo_improvement=epochs_wo_improvement, early_stopping_patience=early_stopping_patience,
                           epochs=epochs, csv_stats=csv_stats, target_dir=target_dir, filename_model=filename_model,
                           register_at_wandb=register_at_wandb,
                           additional_wandb_stats=additional_wandb_stats, test_loader=test_loaders[0], valtest=valtest,
                           test_with_count_edges_msg_aggr=test_with_count_edges_msg_aggr)

            epoch += 1

            if pretrain_lr is not None and epoch > epochs * 0.75 and optimizer.param_groups[0]['lr'] == initial_lr:
                # manually decrease lr if epoch reaches 75% of total epochs and lr has not been reduced yet
                print(f"Manually reducing learning rate to {initial_lr / 10}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr / 10

        except StopIteration:
            epoch += 1
            break

    print(f'Start Finetuning on dataset with UDFs only')
    epoch_offset = epoch
    while epoch < epoch_offset + ft_epochs_udf_only and not finished and not skip_train:
        try:
            train_epoch_fn(epoch=epoch, train_loader=train_loader_udf_only, val_loader=val_loader_udf_only,
                           model=model, optimizer=optimizer, max_epoch_tuples=max_epoch_tuples, prof=prof,
                           apply_gradient_norm=apply_gradient_norm, metrics=metrics, lr_scheduler=ft_lr,
                           epochs_wo_improvement=epochs_wo_improvement, early_stopping_patience=early_stopping_patience,
                           epochs=epoch_offset + ft_epochs_udf_only, csv_stats=csv_stats, target_dir=target_dir,
                           filename_model=filename_model,
                           register_at_wandb=register_at_wandb,
                           additional_wandb_stats=additional_wandb_stats, test_loader=test_loaders[0], valtest=valtest,
                           test_with_count_edges_msg_aggr=test_with_count_edges_msg_aggr)
            epoch += 1

            if ft_lr is not None and epoch > (epoch_offset + ft_epochs_udf_only) * 0.75 and optimizer.param_groups[0][
                'lr'] == initial_lr:
                # manually decrease lr if epoch reaches 75% of total epochs and lr has not been reduced yet
                print(f"Manually reducing learning rate to {initial_lr / 10}")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr / 10

        except StopIteration:
            break

    if watched:
        wandb.unwatch(model)
    print(f'Training finished', flush=True)

    print("Reloading best model")
    early_stop_m = find_early_stopping_metric(metrics)
    best_model_state = early_stop_m.best_model
    if zs_paper_dataset:
        best_model_state = translate_zs_paper_model_state_dict(early_stop_m.best_model)
    model.load_state_dict(best_model_state)

    # apply_pca_evaluation = False

    # evaluate test set
    if test_loaders is not None:
        if not (target_dir is None or filename_model is None):
            # compile path for artifacts
            artifacts_out_path = os.path.join(target_dir, f'{filename_model}_artifacts')
            os.makedirs(artifacts_out_path, exist_ok=True)

            if apply_pca_evaluation:
                print(f'Running inference for train and val set to perform PCA analysis')
                train_labels, train_preds, train_graph_reprs, train_udf_reprs, _, _, _, _, train_query_stats = run_inference(
                    train_loader, model, 5000)
                val_labels, val_preds, val_graph_reprs, val_udf_reprs, _, _, _, _, val_query_stats = run_inference(
                    val_loader, model, 5000)

            else:
                train_labels, train_preds, val_labels, val_preds = None, None, None, None
                train_graph_reprs, train_udf_reprs, val_graph_reprs, val_udf_reprs = None, None, None, None
                train_query_stats, val_query_stats = None, None

            assert len(test_loaders) == len(test_loader_names), f'{len(test_loaders)} vs {len(test_loader_names)}'

            if apply_pca_evaluation:
                if model.return_graph_repr:
                    train_graph_pca = perform_pca_analysis(train_graph_reprs, train_labels, train_preds, label='graph',
                                                           prefix='train', title='train - graph',
                                                           save_path=artifacts_out_path,
                                                           query_stats=train_query_stats)
                    wandb.log(train_graph_pca)

                if model.return_udf_repr:
                    train_udf_pca = perform_pca_analysis(train_udf_reprs, train_labels, train_preds, label='udf',
                                                         prefix='train', title='train - udf',
                                                         save_path=artifacts_out_path,
                                                         query_stats=train_query_stats)
                    wandb.log(train_udf_pca)

            for card in ['est', 'act', 'dd', 'wj']:
                _, _, _, _, _, _, test_loaders, test_loader_names, _ = \
                    create_dataloader([], test_workload_runs, statistics_file, featurization, database,
                                      val_ratio=0.15, finetune_ratio=0.0, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=False, limit_queries=limit_queries,
                                      limit_queries_affected_wl=limit_queries_affected_wl,
                                      loss_class_name=loss_class_name,
                                      offset_np_import=offset_np_import,
                                      stratify_dataset_by_runtimes=stratify_dataset_by_runtimes,
                                      stratify_per_database_by_runtimes=stratify_per_database_by_runtimes,
                                      max_runtime=max_runtime, multi_label_keep_duplicates=multi_label_keep_duplicates,
                                      zs_paper_dataset=zs_paper_dataset,
                                      train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
                                      w_loop_end_node=w_loop_end_node, add_loop_loopend_edge=add_loop_loopend_edge,
                                      card_est_assume_lazy_eval=card_est_assume_lazy_eval,
                                      min_runtime_ms=min_runtime_ms,
                                      card_type_in_udf=card, card_type_above_udf=card,
                                      card_type_below_udf=card, skip_udf=skip_udf,
                                      stratification_prioritize_loops=stratification_prioritize_loops,
                                      plans_have_no_udf=plans_have_no_udf, filter_plans=filter_plans)

                for test_loader, path in zip(test_loaders, test_loader_names):

                    # assemble output paths
                    test_workload_name, test_csv_path = extract_test_paths(path, card=card, target_dir=target_dir,
                                                                           filename_model=filename_model)

                    print(f"Starting validation for {test_workload_name}")

                    test_stats = dict()
                    try:
                        _, wandb_plots, test_graph_reprs, test_udf_reprs, test_labels, test_preds, test_query_stats = validate_model(
                            test_loader,
                            model,
                            epoch=epoch,
                            validate_stats=test_stats,
                            metrics=metrics,
                            log_all_queries=True,
                            extended_evaluation=True,
                            prefix=f'test_{test_workload_name}',
                            is_test_loader=True, log_to_wandb=register_at_wandb)
                    except KeyError as e:
                        print(e)
                        print(f"Error with test loader: {test_workload_name}", flush=True)
                        continue

                    if apply_pca_evaluation:
                        if model.return_graph_repr:
                            wandb_plots.update(
                                perform_pca_analysis(test_graph_reprs, test_labels, test_preds, label='graph',
                                                     prefix=f'test_{test_workload_name}',
                                                     title=f'test_{test_workload_name} - graph',
                                                     save_path=artifacts_out_path,
                                                     query_stats=test_query_stats))
                            wandb_plots.update(
                                perform_pca_analysis(test_graph_reprs, test_labels, test_preds, label='graph',
                                                     prefix=f'test_{test_workload_name}(train)',
                                                     fit_repr=train_graph_reprs,
                                                     title=f'test_{test_workload_name}(train) - graph',
                                                     save_path=artifacts_out_path,
                                                     query_stats=test_query_stats))

                        if model.return_udf_repr:
                            wandb_plots.update(
                                perform_pca_analysis(test_udf_reprs, test_labels, test_preds, label='udf',
                                                     prefix=f'test_{test_workload_name}',
                                                     title=f'test_{test_workload_name} - udf',
                                                     save_path=artifacts_out_path,
                                                     query_stats=test_query_stats))
                            wandb_plots.update(
                                perform_pca_analysis(test_udf_reprs, test_labels, test_preds, label='udf',
                                                     prefix=f'test_{test_workload_name}(train)',
                                                     fit_repr=train_udf_reprs,
                                                     title=f'test_{test_workload_name}(train) - udf',
                                                     save_path=artifacts_out_path,
                                                     query_stats=test_query_stats))

                    if register_at_wandb:
                        # log results to wandb
                        wandb.summary[test_workload_name] = test_stats
                        wandb.log(wandb_plots)

                    save_csv([test_stats], test_csv_path)

                    del test_loader

        else:
            print("Skipping saving the test stats", flush=True)
    else:
        print("No test set evaluation", flush=True)

    # try to delete orphan processes earlier - this should not be necessary but I have severe issues with that...
    def del_loader(loader):
        if isinstance(loader._get_iterator(), _MultiProcessingDataLoaderIter):
            loader._get_iterator()._shutdown_workers()
        del loader

    if train_loader is not None:
        del_loader(train_loader)
    if val_loader is not None:
        del_loader(val_loader)
    for test_loader in test_loaders:
        del_loader(test_loader)

    print(f'Deleted loaders', flush=True)
    torch.cuda.empty_cache()

    if pt_profile:
        print(f'Stop profiler...')
        prof.stop()


def train_epoch_fn(epoch: int, train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader, model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer, max_epoch_tuples: int, prof: torch.profiler.profile,
                   apply_gradient_norm: bool, metrics: List, lr_scheduler: Any, epochs_wo_improvement: int,
                   early_stopping_patience: int, epochs: int, csv_stats: List, target_dir: str, filename_model: str,
                   register_at_wandb: bool, additional_wandb_stats: Dict = None,
                   test_loader: torch.utils.data.DataLoader = None, apply_pca_evaluation: bool = True,
                   valtest: bool = True, test_with_count_edges_msg_aggr: bool = False):
    print(f"Epoch {epoch}")
    epoch_stats = {
        'epoch': epoch,
        'lr': optimizer.param_groups[0]['lr'],
    }

    epoch_start_time = time.perf_counter()

    train_epoch(epoch_stats, train_loader, model, optimizer, max_epoch_tuples, pt_profiler=prof,
                gradient_norm=apply_gradient_norm, test_with_count_edges_msg_aggr=test_with_count_edges_msg_aggr)

    any_best_metric, wandb_plots, val_graph_reprs, val_udf_reprs, val_labels, val_preds, val_query_stats = validate_model(
        val_loader,
        model,
        epoch=epoch,
        validate_stats=epoch_stats,
        metrics=metrics,
        max_epoch_tuples=max_epoch_tuples,
        prefix='val',
        log_to_wandb=register_at_wandb)
    if test_loader is not None and valtest:
        _, valtest_wandb_plots, valtest_graph_reprs, valtest_udf_reprs, valtest_labels, valtest_preds, valtest_query_stats = validate_model(
            test_loader, model, epoch=epoch, validate_stats=epoch_stats,
            metrics=metrics, max_epoch_tuples=max_epoch_tuples,
            prefix='valtest', is_test_loader=True, log_to_wandb=register_at_wandb)
        wandb_plots.update(valtest_wandb_plots)
    else:
        valtest_graph_reprs, valtest_udf_reprs, valtest_labels, valtest_preds, valtest_query_stats = None, None, None, None, None

    plot_out_path = os.path.join(target_dir, f'{filename_model}_artifacts')

    if apply_pca_evaluation:
        os.makedirs(plot_out_path, exist_ok=True)
        # print(plot_out_path)
        if model.return_graph_repr:
            if test_loader is not None and valtest:
                perform_pca_analysis(valtest_graph_reprs, valtest_labels, valtest_preds, label='graph',
                                     prefix=f'test_{epoch}', title=f'test - graph (epoch: {epoch})',
                                     save_path=plot_out_path, return_wandb_image_dict=False,
                                     query_stats=valtest_query_stats)

                perform_pca_analysis(valtest_graph_reprs, valtest_labels, valtest_preds, label='graph',
                                     prefix=f'testval_{epoch}', fit_repr=val_graph_reprs,
                                     title=f'test(val) - graph (epoch: {epoch})', save_path=plot_out_path,
                                     return_wandb_image_dict=False, query_stats=valtest_query_stats)

            perform_pca_analysis(val_graph_reprs, val_labels, val_preds, label='graph',
                                 prefix=f'val_{epoch}', title=f'val - graph (epoch: {epoch})',
                                 save_path=plot_out_path,
                                 return_wandb_image_dict=False, query_stats=val_query_stats)

        if model.return_udf_repr:
            if test_loader is not None and valtest:
                perform_pca_analysis(valtest_udf_reprs, valtest_labels, valtest_preds, label='udf',
                                     prefix=f'test_{epoch}', title=f'test - udf (epoch: {epoch})',
                                     save_path=plot_out_path,
                                     return_wandb_image_dict=False, query_stats=valtest_query_stats)

                perform_pca_analysis(valtest_udf_reprs, valtest_labels, valtest_preds, label='udf',
                                     prefix=f'testval_{epoch}', fit_repr=val_udf_reprs,
                                     title=f'test(val) - udf (epoch: {epoch})',
                                     save_path=plot_out_path, return_wandb_image_dict=False,
                                     query_stats=valtest_query_stats)

            perform_pca_analysis(val_udf_reprs, val_labels, val_preds, label='udf',
                                 prefix=f'val_{epoch}', title=f'val - udf (epoch: {epoch})',
                                 save_path=plot_out_path,
                                 return_wandb_image_dict=False, query_stats=val_query_stats)

    epoch_stats.update(epoch=epoch, epoch_time=time.perf_counter() - epoch_start_time)

    if lr_scheduler is not None:
        lr_scheduler.step(epoch_stats['val_loss'])

    # see if we can already stop the training
    stop_early = False
    if not any_best_metric:
        epochs_wo_improvement += 1
        if early_stopping_patience is not None and epochs_wo_improvement > early_stopping_patience:
            stop_early = True
    else:
        epochs_wo_improvement = 0

    # also set finished to true if this is the last epoch
    if epoch == epochs - 1:
        stop_early = True

    epoch_stats.update(stop_early=stop_early)
    print(f"epochs_wo_improvement: {epochs_wo_improvement}")

    if register_at_wandb:
        # log stats to wand
        stats = copy(epoch_stats)
        if additional_wandb_stats is not None:
            stats.update(additional_wandb_stats)
        if wandb_plots is not None:
            stats.update(wandb_plots)
        wandb.log(stats)

    # save stats to file
    csv_stats.append(epoch_stats)

    # save current state of training allowing us to resume if this is stopped
    save_checkpoint(epochs_wo_improvement, epoch, model, optimizer, lr_scheduler=lr_scheduler,
                    target_path=target_dir,
                    model_name=filename_model, metrics=metrics, csv_stats=csv_stats, finished=stop_early)

    epoch += 1

    if stop_early:
        if epoch == epochs - 1:
            print(f"Finished training after {epoch} epochs")
        else:
            print(f"Early stopping kicked in due to no improvement in {early_stopping_patience} epochs")
        raise StopIteration()


def optuna_intermediate_value(metrics):
    for m in metrics:
        if m.early_stopping_metric:
            assert isinstance(m, QError)
            return m.best_seen_value
    raise ValueError('Metric invalid')
