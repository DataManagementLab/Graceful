import argparse
import functools
from collections import defaultdict
from typing import Optional

import numpy as np
import wandb

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.utils import load_json
from evaluate_pull_up_predictor import log_q_errors, convert_dict_of_lists_to_dataframe
from hyperparams_utils import get_config
from models.dataset.dataset_creation import read_workload_runs, create_datasets, create_dataloader, \
    derive_label_normalizer
from models.training.checkpoint import load_checkpoint
from models.training.metrics import RMSE, QError, MAPE
from models.training.train import run_inference
from models.training.utils import find_early_stopping_metric
from models.zero_shot_models.specific_models.model import zero_shot_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plans_path', required=True, default=None)
    parser.add_argument('--model_dir', required=True, default=None)
    parser.add_argument('--model_name', required=True, default=None)
    parser.add_argument('--model_config', required=True, default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--statistics_file', required=True, default=None)
    parser.add_argument('--data_keyword', default='complex_dd')
    parser.add_argument('--dataset', required=True, default=None)

    ###
    # Begin Model Args
    ###
    parser.add_argument('--mp_ignore_udf', type=bool, default=argparse.SUPPRESS)
    parser.add_argument('--work_with_udf_repr', default=False, action='store_true')
    ###
    # End Args
    ###

    args = parser.parse_args()

    wandb.init(
        project='udf_cost_est',
        entity='jwehrstein',
        name=f'{args.dataset}_{args.model_config}',
        config=args.__dict__,
    )

    #######
    database = DatabaseSystem.DUCKDB
    #######

    # parse
    plans_path = args.plans_path
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

    # create dataset
    plans, dataset_stats = read_workload_runs([plans_path], min_runtime_ms=config['min_runtime_ms'],
                                              max_runtime=config['max_runtime'])

    sql_list = [plan.query for plan in plans]

    _, dataset, _, _, _, database_statistics = create_datasets(None,
                                                               loss_class_name=None,
                                                               val_ratio=0,
                                                               shuffle_before_split=False,
                                                               stratify_dataset_by_runtimes=False,
                                                               # avoid double stratification. Perform only once since we only have one database anyways here
                                                               max_runtime=config['max_runtime'],
                                                               zs_paper_dataset=False,
                                                               train_udf_graph_against_udf_runtime=False,
                                                               min_runtime_ms=config['min_runtime_ms'],
                                                               infuse_plans=plans,
                                                               infuse_database_statistics=dataset_stats)

    create_dataset_fn_test_artefacts = {
        plans_path: (_, dataset, _, database_statistics),
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
    preds_dict = defaultdict(dict)
    qerrors = defaultdict(dict)


    def run_inference_fn(card_type_below_udf: str, card_type_in_udf: str, card_type_above_udf: str,
                         card_est_udf_sel: Optional[int], expected_labels=None, ):

        # assemble dataloaders
        _, _, _, _, _, _, data_loaders, _, _ = create_dataloder_fn(featurization=config['featurization'],
                                                                   est_card_udf_sel=card_est_udf_sel,
                                                                   feature_statistics=feature_statistics,
                                                                   card_type_below_udf=card_type_below_udf,
                                                                   card_type_in_udf=card_type_in_udf,
                                                                   card_type_above_udf=card_type_above_udf,
                                                                   test_workload_run_paths=[plans_path],
                                                                   )

        # assemble card ids
        if card_type_below_udf == card_type_in_udf == card_type_above_udf:
            card_id = card_type_below_udf
        else:
            card_id = f'{card_type_below_udf}_udf{card_type_in_udf}_{card_type_above_udf}'

        if len(data_loaders) == 0:
            return None, None

        # run inference for pull-up plans
        rcv_labels, preds, graph_reprs, udf_reprs, sample_idxs, val_num_tuples, test_start_t, val_loss, stats = run_inference(
            data_loaders[0], model,
            100000)

        if expected_labels is not None:
            assert np.all(
                rcv_labels == expected_labels), f'Pullup labels do not match expected labels\n{rcv_labels}\n{expected_labels}'
        assert np.all(rcv_labels >= 0.05), f'Pullup labels should be greater than 0.05\n{rcv_labels}'

        print(f'{card_est_udf_sel} ({card_id})')
        qerrors[card_id][card_est_udf_sel] = log_q_errors(preds, rcv_labels)

        preds_dict[card_id][card_est_udf_sel] = preds

        return rcv_labels


    labels = run_inference_fn('act', 'act', 'act', None)
    run_inference_fn('dd', 'dd', 'dd', None, expected_labels=labels)

    log_dict = {
        'labels': wandb.Table(
            dataframe=convert_dict_of_lists_to_dataframe(
                {'labels': labels, 'sql': sql_list,
                 })),
    }

    for key, val in preds_dict.items():
        log_dict[f'preds_{key}'] = wandb.Table(dataframe=convert_dict_of_lists_to_dataframe(val))

    wandb.log(log_dict)
    print('Done')
