import argparse
import functools
from collections import defaultdict
from typing import Optional

import numpy as np
import wandb

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.utils import load_json
from models.dataset.dataset_creation import create_dataloader, derive_label_normalizer
from models.training.checkpoint import load_checkpoint
from models.training.metrics import QError, RMSE, MAPE
from models.training.train import run_inference
from models.training.utils import find_early_stopping_metric
from models.zero_shot_models.specific_models.model import zero_shot_models
from pull_push_advisor.eval_advisor import run_plan_advisor
from pull_push_advisor.utils import assemble_pullup_pushdown_overlap_datasets, log_q_errors, \
    create_pull_push_label_plot, convert_dict_of_lists_to_dataframe, gen_qerror_plot
from utils.hyperparams_utils import get_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pullup_plans_path', required=True, default=None)
    parser.add_argument('--pushdown_plans_path', required=True, default=None)
    parser.add_argument('--model_dir', required=True, default=None)
    parser.add_argument('--model_name', required=True, default=None)
    parser.add_argument('--model_config', required=True, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--statistics_file', required=True, default=None)
    parser.add_argument('--data_keyword', default='complex_dd')
    parser.add_argument('--dataset', required=True, default=None)
    parser.add_argument('--wandb', default=False, action='store_true')

    ###
    # Begin Model Args
    ###
    parser.add_argument('--mp_ignore_udf', type=bool, default=argparse.SUPPRESS)
    parser.add_argument('--work_with_udf_repr', default=False, action='store_true')
    ###
    # End Args
    ###


    args = parser.parse_args()
    log_to_wandb = args.wandb

    if log_to_wandb:
        wandb.init(
            project='udf_pullup_predictor',
            entity='jwehrstein',
            name=f'{args.dataset}_{args.model_config}',
            config=args.__dict__,
        )

    #######
    sel_list = [10, 30, 50, 70, 90, None]
    database = DatabaseSystem.DUCKDB
    #######

    # parse
    pullup_plans_path = args.pullup_plans_path
    pushdown_plans_path = args.pushdown_plans_path
    statistics_file = args.statistics_file
    model_name = args.model_name
    model_dir = args.model_dir
    device = args.device
    model_config = args.model_config

    args_config = {
        'model_config': model_config,
        'data_keyword': args.data_keyword
    }

    checkpoint_map_location = {'cuda:1': device, 'cuda:0': device, 'cpu': device}

    if hasattr(args, 'mp_ignore_udf'):
        args_config['mp_ignore_udf'] = args.mp_ignore_udf
    if args.work_with_udf_repr:
        args_config['work_with_udf_repr'] = args.work_with_udf_repr

    orig_args_config = args_config.copy()
    config, _, _, _, _ = get_config(args_config, wl_base_path='', assemble_filenames=False)

    pullup_test_dataset, pullup_test_database_statistics, pushdown_test_dataset, pushdown_test_database_statistics, pushdown_sql, pullup_sql = assemble_pullup_pushdown_overlap_datasets(
        pullup_plans_path, pushdown_plans_path, min_runtime_ms=config['min_runtime_ms'],
        max_runtime=config['max_runtime'])

    create_dataset_fn_test_artefacts = {
        pullup_plans_path: (_, pullup_test_dataset, _, pullup_test_database_statistics),
        pushdown_plans_path: (_, pushdown_test_dataset, _, pushdown_test_database_statistics)
    }

    # create data loader
    create_dataloder_fn = functools.partial(create_dataloader, workload_run_paths=[],
                                            statistics_file=statistics_file,
                                            database=database,
                                            val_ratio=0.15, finetune_ratio=0.0, batch_size=config['batch_size'],
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=False, limit_queries=False,
                                            limit_queries_affected_wl=None,
                                            loss_class_name=config['final_mlp_kwargs']['loss_class_name'],
                                            offset_np_import=config['offset_np_import'],
                                            stratify_dataset_by_runtimes=config['stratify_dataset_by_runtimes'],
                                            stratify_per_database_by_runtimes=config[
                                                'stratify_per_database_by_runtimes'],
                                            max_runtime=config['max_runtime'],
                                            multi_label_keep_duplicates=config['multi_label_keep_duplicates'],
                                            zs_paper_dataset=config['zs_paper_dataset'],
                                            train_udf_graph_against_udf_runtime=config[
                                                'train_udf_graph_against_udf_runtime'],
                                            w_loop_end_node=config['w_loop_end_node'],
                                            add_loop_loopend_edge=config['add_loop_loopend_edge'],
                                            card_est_assume_lazy_eval=config['card_est_assume_lazy_eval'],
                                            min_runtime_ms=config['min_runtime_ms'],
                                            create_dataset_fn_test_artefacts=create_dataset_fn_test_artefacts)

    feature_statistics = load_json(statistics_file, namespace=False)
    # add stats for artificial features (additional flags / ...)
    feature_statistics['on_udf'] = {"value_dict": {"True": 0, "False": 1}, "no_vals": 2, "type": "categorical"}

    label_norm = derive_label_normalizer('QLoss', np.asarray([1, 10, 10]))

    # create zero shot model dependent on database
    model = zero_shot_models[database](device=device, final_mlp_kwargs=config['final_mlp_kwargs'],
                                       node_type_kwargs=config['node_type_kwargs'], output_dim=1,
                                       feature_statistics=feature_statistics,
                                       tree_layer_kwargs=config['tree_layer_kwargs'],
                                       featurization=config['featurization'],
                                       label_norm=label_norm, mp_ignore_udf=config['mp_ignore_udf'],
                                       return_graph_repr=True,
                                       return_udf_repr=True,
                                       plans_have_no_udf=False,
                                       train_udf_graph_against_udf_runtime=False,
                                       work_with_udf_repr=config['work_with_udf_repr'],
                                       test_with_count_edges_msg_aggr=False)

    # move to gpu
    model = model.to(model.device)

    metrics = [RMSE(), MAPE(), QError(percentile=50, early_stopping_metric=True), QError(percentile=95),
               QError(percentile=100)]

    # load checkpoint
    csv_stats, epochs_wo_improvement, epoch, model, optimizer, lr_scheduler, metrics, finished = \
        load_checkpoint(model, model_dir, model_name, optimizer=None,
                        lr_scheduler=None,
                        metrics=metrics, filetype='.pt', zs_paper_model=False, map_location=checkpoint_map_location)

    # reloading best model
    early_stop_m = find_early_stopping_metric(metrics)
    best_model_state = early_stop_m.best_model
    model.load_state_dict(best_model_state)

    # run inference for different udf filter selectivity assumptions
    preds_dict_pullup = defaultdict(dict)
    preds_dict_pushdown = defaultdict(dict)
    pullup_qerrors = defaultdict(dict)
    pushdown_qerrors = defaultdict(dict)


    def run_inference_fn(pullup_card_type_below_udf: str, pullup_card_type_in_udf: str, pullup_card_type_above_udf: str,
                         card_est_udf_sel: Optional[int],
                         pushdown_card_type_below_udf: str = None, pushdown_card_type_in_udf: str = None,
                         pushdown_card_type_above_udf: str = None,
                         expected_pullup_labels=None, expected_pushdown_labels=None):
        assert pullup_card_type_below_udf is not None

        if pushdown_card_type_below_udf is None:
            pushdown_card_type_below_udf = pullup_card_type_below_udf
        if pushdown_card_type_in_udf is None:
            pushdown_card_type_in_udf = pullup_card_type_in_udf
        if pushdown_card_type_above_udf is None:
            pushdown_card_type_above_udf = pullup_card_type_above_udf

        # assemble dataloaders
        _, _, _, _, _, _, pullup_data_loaders, _, _ = create_dataloder_fn(featurization=config['featurization'],
                                                                          est_card_udf_sel=card_est_udf_sel,
                                                                          feature_statistics=feature_statistics,
                                                                          card_type_below_udf=pullup_card_type_below_udf,
                                                                          card_type_in_udf=pullup_card_type_in_udf,
                                                                          card_type_above_udf=pullup_card_type_above_udf,
                                                                          test_workload_run_paths=[pullup_plans_path],
                                                                          )
        _, _, _, _, _, _, pushdown_data_loaders, _, _ = create_dataloder_fn(featurization=config['featurization'],
                                                                            est_card_udf_sel=card_est_udf_sel,
                                                                            feature_statistics=feature_statistics,
                                                                            card_type_below_udf=pushdown_card_type_below_udf,
                                                                            card_type_in_udf=pushdown_card_type_in_udf,
                                                                            card_type_above_udf=pushdown_card_type_above_udf,
                                                                            test_workload_run_paths=[
                                                                                pushdown_plans_path],
                                                                            )

        # assemble card ids
        if pushdown_card_type_below_udf == pushdown_card_type_in_udf == pushdown_card_type_above_udf:
            pushdown_card_id = pushdown_card_type_below_udf
        else:
            pushdown_card_id = f'{pushdown_card_type_below_udf}_udf{pushdown_card_type_in_udf}_{pushdown_card_type_above_udf}'

        if pullup_card_type_below_udf == pullup_card_type_in_udf == pullup_card_type_above_udf:
            pullup_card_id = pullup_card_type_below_udf
        else:
            pullup_card_id = f'{pullup_card_type_below_udf}_udf{pullup_card_type_in_udf}_{pullup_card_type_above_udf}'

        if len(pullup_data_loaders) == 0 or len(pushdown_data_loaders) == 0:
            return None, None

        # run inference for pull-up plans
        rcv_pullup_labels, pullup_preds, pullup_graph_reprs, pullup_udf_reprs, sample_idxs, val_num_tuples, test_start_t, val_loss, stats = run_inference(
            pullup_data_loaders[0], model,
            100000)

        if expected_pullup_labels is not None:
            assert np.all(
                rcv_pullup_labels == expected_pullup_labels), f'Pullup labels do not match expected labels\n{rcv_pullup_labels}\n{expected_pullup_labels}'
        assert np.all(rcv_pullup_labels >= 0.05), f'Pullup labels should be greater than 0.05\n{rcv_pullup_labels}'

        print(f'Pullup {card_est_udf_sel} ({pullup_card_id})')
        pullup_qerrors[pullup_card_id][card_est_udf_sel] = log_q_errors(pullup_preds, rcv_pullup_labels)

        # run inference for push-down plans
        rcv_pushdown_labels, pushdown_preds, pushdown_graph_reprs, pushdown_udf_reprs, sample_idxs, val_num_tuples, test_start_t, val_loss, stats = run_inference(
            pushdown_data_loaders[0], model,
            100000)

        if expected_pushdown_labels is not None:
            assert np.all(
                rcv_pushdown_labels == expected_pushdown_labels), f'Pushdown labels do not match expected labels\n{rcv_pushdown_labels}\n{expected_pushdown_labels}'
        assert np.all(
            rcv_pushdown_labels >= 0.05), f'Pushdown labels should be greater than 0.05\n{rcv_pushdown_labels}'

        print(f'Pushdown {card_est_udf_sel} ({pushdown_card_id})')
        pushdown_qerrors[pushdown_card_id][card_est_udf_sel] = log_q_errors(pushdown_preds, rcv_pushdown_labels)

        assert len(rcv_pushdown_labels) == len(rcv_pullup_labels)

        preds_dict_pullup[pullup_card_id][card_est_udf_sel] = pullup_preds
        preds_dict_pushdown[pushdown_card_id][card_est_udf_sel] = pushdown_preds

        return rcv_pullup_labels, rcv_pushdown_labels


    pullup_labels, pushdown_labels = run_inference_fn('act', 'act', 'act', None)

    for card_udf_sel in sel_list:
        run_inference_fn(pullup_card_type_below_udf='est', pullup_card_type_in_udf='est',
                         pullup_card_type_above_udf='est',
                         card_est_udf_sel=card_udf_sel, expected_pullup_labels=pullup_labels,
                         expected_pushdown_labels=pushdown_labels)
        run_inference_fn(pullup_card_type_below_udf='dd', pullup_card_type_in_udf='dd', pullup_card_type_above_udf='dd',
                         card_est_udf_sel=card_udf_sel, expected_pullup_labels=pullup_labels,
                         expected_pushdown_labels=pushdown_labels)
        # run_inference_fn(pullup_card_type_below_udf='wj', pullup_card_type_in_udf='wj', pullup_card_type_above_udf='wj',
        #                  pushdown_card_type_below_udf='dd', pushdown_card_type_in_udf='dd',
        #                  pushdown_card_type_above_udf='dd',
        #                  card_est_udf_sel=card_udf_sel, expected_pullup_labels=pullup_labels,
        #                  expected_pushdown_labels=pushdown_labels)
        run_inference_fn(pullup_card_type_below_udf='wj', pullup_card_type_in_udf='wj', pullup_card_type_above_udf='wj',
                         pushdown_card_type_below_udf='wj', pushdown_card_type_in_udf='wj',
                         pushdown_card_type_above_udf='wj',
                         card_est_udf_sel=card_udf_sel, expected_pullup_labels=pullup_labels,
                         expected_pushdown_labels=pushdown_labels)

    if log_to_wandb:
        # log results to wandb - this allows viewing plots and aggregating results over dataset-runs
        plot = create_pull_push_label_plot(pullup_labels, pushdown_labels)
        wandb.run.log({'labels_plot': wandb.Image(plot)})

        # log to wandb
        log_dict = {
            'labels': wandb.Table(
                dataframe=convert_dict_of_lists_to_dataframe(
                    {'pullup': pullup_labels, 'pushdown': pushdown_labels, 'pullup_sql': pullup_sql,
                     'pushdown_sql': pushdown_sql})),
            'sel_list': sel_list,

        }

        for key, val in preds_dict_pullup.items():
            log_dict[f'preds_pullup_{key}'] = wandb.Table(dataframe=convert_dict_of_lists_to_dataframe(val))
        for key, val in preds_dict_pushdown.items():
            log_dict[f'preds_pushdown_{key}'] = wandb.Table(dataframe=convert_dict_of_lists_to_dataframe(val))

        wandb.log(log_dict)
        plot = gen_qerror_plot(pushdown_qerrors, pullup_qerrors)
        wandb.log({"qerror_chart": wandb.Image(plot)})

    # create dict for offline processing
    offline_dict = {
        'labels': convert_dict_of_lists_to_dataframe(
            {'pullup': pullup_labels, 'pushdown': pushdown_labels, 'pullup_sql': pullup_sql,
             'pushdown_sql': pushdown_sql}),
        'sel_list': sel_list,
    }
    for key, val in preds_dict_pullup.items():
        offline_dict[f'preds_pullup_{key}'] = convert_dict_of_lists_to_dataframe(val)
    for key, val in preds_dict_pushdown.items():
        offline_dict[f'preds_pushdown_{key}'] = convert_dict_of_lists_to_dataframe(val)

    # run plan advisor
    run_plan_advisor(offline_dict, args.dataset)
