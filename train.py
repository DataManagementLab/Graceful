import argparse
import copy
import datetime
import functools
import os
import random
from typing import Dict, Any

try:
    import wandb
except:
    pass

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from models.training.train import train_model
from utils.hyperparams_utils import get_config


def run_train(
        orig_args_config: Dict[str, Any],

        wl_base_path: str,
        out_base_path: str,

        device: str,
        register_at_wandb: bool,
        seed: int = 0,
        wandb_run_data: Dict = None,

        limit_queries: int = None,
        limit_queries_affected_wl: int = None,
        database: DatabaseSystem = DatabaseSystem.POSTGRES,
        num_workers: int = 1,
        max_epoch_tuples: int = 100000,
        skip_train: bool = False,
        pt_profile: bool = False,
        apply_pca_evaluation: bool = False,
        test_only: bool = False,

):
    # make a copy of the config dict, so that a subsequent sweep run does not read the modified config
    args_config = copy.deepcopy(orig_args_config)

    if register_at_wandb:
        if wandb_run_data['sweep']:
            wandb.init(tags=['udf_cost'])
            # generate filename with wandb
            args_config.update(wandb.config.as_dict())

    print(f'Running with config: {args_config}')

    config, train_wl_paths, test_wl_paths, statistics_file, model_name = get_config(args_config,
                                                                                    wl_base_path=wl_base_path, )

    print(config)

    if register_at_wandb and not wandb_run_data['sweep']:
        assert args_config is not None
        wandb.init(
            project=wandb_run_data['project'],
            entity=wandb_run_data['entity'],
            name=model_name,
            group=wandb_run_data['group'],
            config=orig_args_config,
            id=wandb_run_data['id'],
            resume='allow',
            tags=['zs_cost_partial']
        )

    model_dirname = model_name
    # create model name prefixed with date, time and random value
    model_filename = f'{model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{random.randint(0, 100):03d}'

    model_dir_path = os.path.join(out_base_path, model_dirname)

    train_model(workload_runs=train_wl_paths,
                test_workload_runs=test_wl_paths,
                statistics_file=statistics_file,
                target_dir=model_dir_path,
                filename_model=model_filename,
                output_dim=1,
                device=device,
                max_epoch_tuples=max_epoch_tuples,
                num_workers=num_workers,
                database=database,
                limit_queries=limit_queries,
                limit_queries_affected_wl=limit_queries_affected_wl,
                seed=seed,
                skip_train=skip_train,
                register_at_wandb=register_at_wandb,
                pt_profile=pt_profile,
                apply_pca_evaluation=apply_pca_evaluation, test_only=test_only,
                **config,
                include_no_udf_data=orig_args_config[
                    'include_no_udf_data'] if 'include_no_udf_data' in orig_args_config else False,
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--wl_base_path', type=str, required=True)
    parser.add_argument('--out_base_path', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--data_keyword', type=str, required=True)

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--max_epoch_tuples', type=int, default=100000)
    parser.add_argument('--limit_queries', type=int, default=None)
    parser.add_argument('--limit_queries_affected_wl', type=int, default=None)
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--pt_profile', action='store_true')
    parser.add_argument('--max_runtime', default=30, type=int,
                        help='max runtime in seconds, used for balancing dataset by runtimes')

    parser.add_argument('--apply_pca_evaluation', default=False, action='store_true', )
    parser.add_argument('--test_only', default=False, action='store_true', )

    # wandb related stuff
    parser.add_argument('--register_at_wandb', default=False, action='store_true')
    parser.add_argument('--wandb_run_sweep', default=False, action='store_true')
    parser.add_argument('--wandb_sweep_id', default=None)
    parser.add_argument('--wandb_resume_id', default=None, type=str)
    parser.add_argument('--wandb_project', default='GRACEFUL', type=str)
    parser.add_argument('--wandb_entity', default='', type=str)

    # optional config
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--epochs', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--ft_epochs_udf_only', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--early_stopping_patience', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--test_against', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--train_on_test', type=bool, default=argparse.SUPPRESS)
    parser.add_argument('--stratify_dataset_by_runtimes', type=bool, default=argparse.SUPPRESS)
    parser.add_argument('--stratify_per_database_by_runtimes', type=bool, default=argparse.SUPPRESS)
    parser.add_argument('--stratification_prioritize_loops', type=bool, default=argparse.SUPPRESS)
    parser.add_argument('--mp_ignore_udf', type=bool, default=argparse.SUPPRESS)
    parser.add_argument('--optimizer', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--min_runtime_ms', default=argparse.SUPPRESS, type=int,
                        help='min runtime in ms for plans to consider')

    parser.add_argument('--zs_paper_dataset', default=False, action='store_true')
    parser.add_argument('--plans_have_no_udf', default=False, action='store_true')
    parser.add_argument('--train_udf_graph_against_udf_runtime', default=False, action='store_true')
    parser.add_argument('--work_with_udf_repr', default=False, action='store_true')
    parser.add_argument('--include_no_udf_data', default=False, action='store_true')
    parser.add_argument('--include_pullup_data', default=False, action='store_true')
    parser.add_argument('--include_pushdown_data', default=False, action='store_true')
    parser.add_argument('--include_no_udf_data_large', default=False, action='store_true')
    parser.add_argument('--include_select_only_w_branch', default=False, action='store_true')

    parser.add_argument('--skip_udf', type=bool, default=argparse.SUPPRESS)

    parser.add_argument('--test_with_count_edges_msg_aggr', default=False, action='store_true')

    parser.add_argument('--pretrained_model_artifact_dir', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--pretrained_model_filename', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--card_type', type=str, default=argparse.SUPPRESS)

    # filter plans
    parser.add_argument('--min_num_branches', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--max_num_branches', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--min_num_loops', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--max_num_loops', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--min_num_np_calls', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--max_num_np_calls', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--min_num_math_calls', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--max_num_math_calls', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--min_num_comp_nodes', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--max_num_comp_nodes', type=int, default=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.register_at_wandb or args.wandb_run_sweep:
        wandb_run_data = {
            'project': args.wandb_project,
            'entity': args.wandb_entity,
            # 'config': wandb_config_data,
        }

        if not args.wandb_run_sweep:
            if args.wandb_resume_id is not None:
                print(f'Resume wandb run: {args.wandb_resume_id}')
            wandb_run_data['id'] = args.wandb_resume_id
            wandb_run_data['group'] = args.data_keyword

        wandb_run_data['sweep'] = args.wandb_run_sweep
    else:
        wandb_run_data = None

    # required args
    args_config = {
        'model_config': args.model_config,  # _ concatenated string of model related keywords
        'data_keyword': args.data_keyword,
        'max_runtime': args.max_runtime,
    }

    # optional args
    if hasattr(args, 'batch_size'):
        args_config['batch_size'] = args.batch_size
    if hasattr(args, 'epochs'):
        args_config['epochs'] = args.epochs
    if hasattr(args, 'ft_epochs_udf_only'):
        args_config['ft_epochs_udf_only'] = args.ft_epochs_udf_only
    if hasattr(args, 'early_stopping_patience'):
        args_config['early_stopping_patience'] = args.early_stopping_patience
    if hasattr(args, 'test_against'):
        args_config['test_against'] = args.test_against
    if hasattr(args, 'train_on_test'):
        args_config['train_on_test'] = args.train_on_test
    if hasattr(args, 'stratify_dataset_by_runtimes'):
        args_config['stratify_dataset_by_runtimes'] = args.stratify_dataset_by_runtimes
    if hasattr(args, 'stratify_per_database_by_runtimes'):
        args_config['stratify_per_database_by_runtimes'] = args.stratify_per_database_by_runtimes
    if hasattr(args, 'stratification_prioritize_loops'):
        args_config['stratification_prioritize_loops'] = args.stratification_prioritize_loops
    if hasattr(args, 'mp_ignore_udf'):
        args_config['mp_ignore_udf'] = args.mp_ignore_udf
    if hasattr(args, 'optimizer'):
        args_config['optimizer'] = args.optimizer
    if args.zs_paper_dataset:
        args_config['zs_paper_dataset'] = args.zs_paper_dataset
    if args.plans_have_no_udf:
        args_config['plans_have_no_udf'] = args.plans_have_no_udf
    if hasattr(args, 'pretrained_model_artifact_dir'):
        args_config['pretrained_model_artifact_dir'] = args.pretrained_model_artifact_dir
    if hasattr(args, 'pretrained_model_filename'):
        args_config['pretrained_model_filename'] = args.pretrained_model_filename
    if args.train_udf_graph_against_udf_runtime:
        args_config['train_udf_graph_against_udf_runtime'] = args.train_udf_graph_against_udf_runtime
    if args.work_with_udf_repr:
        args_config['work_with_udf_repr'] = args.work_with_udf_repr
    if args.include_no_udf_data:
        args_config['include_no_udf_data'] = args.include_no_udf_data
    if args.include_pullup_data:
        args_config['include_pullup_data'] = args.include_pullup_data
    if args.include_pushdown_data:
        args_config['include_pushdown_data'] = args.include_pushdown_data
    if args.include_no_udf_data_large:
        args_config['include_no_udf_data_large'] = args.include_no_udf_data_large
    if args.include_select_only_w_branch:
        args_config['include_select_only_w_branch'] = args.include_select_only_w_branch
    if args.test_with_count_edges_msg_aggr:
        args_config['test_with_count_edges_msg_aggr'] = args.test_with_count_edges_msg_aggr
    if hasattr(args, 'min_runtime_ms'):
        args_config['min_runtime_ms'] = args.min_runtime_ms
    if hasattr(args, 'card_type'):
        args_config['card_type'] = args.card_type
    if hasattr(args, 'skip_udf'):
        args_config['skip_udf'] = args.skip_udf
    if hasattr(args, 'min_num_branches'):
        args_config['min_num_branches'] = args.min_num_branches
    if hasattr(args, 'max_num_branches'):
        args_config['max_num_branches'] = args.max_num_branches
    if hasattr(args, 'min_num_loops'):
        args_config['min_num_loops'] = args.min_num_loops
    if hasattr(args, 'max_num_loops'):
        args_config['max_num_loops'] = args.max_num_loops
    if hasattr(args, 'min_num_np_calls'):
        args_config['min_num_np_calls'] = args.min_num_np_calls
    if hasattr(args, 'max_num_np_calls'):
        args_config['max_num_np_calls'] = args.max_num_np_calls
    if hasattr(args, 'min_num_math_calls'):
        args_config['min_num_math_calls'] = args.min_num_math_calls
    if hasattr(args, 'max_num_math_calls'):
        args_config['max_num_math_calls'] = args.max_num_math_calls
    if hasattr(args, 'min_num_comp_nodes'):
        args_config['min_num_comp_nodes'] = args.min_num_comp_nodes
    if hasattr(args, 'max_num_comp_nodes'):
        args_config['max_num_comp_nodes'] = args.max_num_comp_nodes

    train_fn = functools.partial(run_train,
                                 orig_args_config=args_config,
                                 wl_base_path=args.wl_base_path,
                                 out_base_path=args.out_base_path,
                                 device=args.device,
                                 register_at_wandb=args.register_at_wandb,
                                 seed=args.seed,
                                 wandb_run_data=wandb_run_data,
                                 database=args.database,
                                 num_workers=args.num_workers,
                                 max_epoch_tuples=args.max_epoch_tuples,
                                 skip_train=args.skip_train,
                                 pt_profile=args.pt_profile,
                                 apply_pca_evaluation=args.apply_pca_evaluation,
                                 test_only=args.test_only,
                                 )

    if args.wandb_run_sweep:
        assert args.wandb_sweep_id is not None
        wandb.agent(args.wandb_sweep_id, function=train_fn, entity=wandb_run_data['entity'],
                    project=wandb_run_data['project'])
    else:
        train_fn()

    print(f'Done!')
