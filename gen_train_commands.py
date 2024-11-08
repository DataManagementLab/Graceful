import argparse

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.datasets.datasets import dataset_list_dict


def gen_cmd(data_keyword: str, model_config: str, artifacts_path: str, wl_base_path: str, device: str = 'cuda:0',
            num_workers: int = None,
            max_epoch_tuples=None, limit_queries: int = None,
            limit_queries_affected_wl: int = None, database: DatabaseSystem = DatabaseSystem.DUCKDB,
            skip_train: bool = False, pt_profile: bool = False, apply_pca_evaluation: bool = True,
            test_only: bool = False, register_at_wandb: bool = True, wandb_run_sweep: bool = False,
            wandb_sweep_id: str = None, wandb_resume_id: str = None, wandb_project: str = None,
            wandb_entity: str = None,
            batch_size: int = None, epochs: int = None,
            ft_epochs_udf_only: int = None,
            early_stopping_patience: int = None, test_against: str = None, train_on_test: bool = None,
            stratify_dataset_by_runtimes: bool = None, stratify_per_database_by_runtimes: bool = None,
            stratification_prioritize_loops: bool = None,
            mp_ignore_udf: bool = None, quick_train: bool = False,
            zs_paper_dataset: bool = False,
            plans_have_no_udf: bool = False, pretrained_model_artifact_dir: str = None,
            pretrained_model_filename: str = None, train_udf_graph_against_udf_runtime: bool = False,
            work_with_udf_repr: bool = False, max_runtime: int = None, include_no_udf_data: bool = False,
            include_pullup_data: bool = False, include_pushdown_data: bool = True,
            include_no_udf_data_large: bool = False, include_select_only_w_branch: bool = False, skip_udf: bool = None,
            test_with_count_edges_msg_aggr: bool = False, min_runtime_ms: int = None, card_type: str = None,
            min_num_branches: int = None, max_num_branches: int = None, min_num_loops: int = None,
            max_num_loops: int = None,
            min_num_np_calls: int = None, max_num_np_calls: int = None, min_num_math_calls: int = None,
            max_num_math_calls: int = None,
            min_num_comp_nodes: int = None, max_num_comp_nodes: int = None):
    out_base_path = artifacts_path

    args_str = ''

    if num_workers is not None:
        args_str += f' --num_workers {num_workers}'
    if max_epoch_tuples is not None:
        args_str += f' --max_epoch_tuples {max_epoch_tuples}'
    if limit_queries is not None:
        args_str += f' --limit_queries {limit_queries}'
    if limit_queries_affected_wl is not None:
        args_str += f' --limit_queries_affected_wl {limit_queries_affected_wl}'
    if database is not None:
        args_str += f' --database {database}'
    if skip_train:
        args_str += f' --skip_train'
    if pt_profile:
        args_str += f' --pt_profile'
    if apply_pca_evaluation:
        args_str += f' --apply_pca_evaluation'
    if test_only:
        args_str += f' --test_only'
    if register_at_wandb:
        args_str += f' --register_at_wandb'
    if wandb_run_sweep:
        args_str += f' --wandb_run_sweep'
    if wandb_sweep_id is not None:
        args_str += f' --wandb_sweep_id {wandb_sweep_id}'
    if wandb_resume_id is not None:
        args_str += f' --wandb_resume_id {wandb_resume_id}'
    if wandb_project is not None:
        args_str += f' --wandb_project {wandb_project}'
    if wandb_entity is not None:
        args_str += f' --wandb_entity {wandb_entity}'
    if batch_size is not None:
        args_str += f' --batch_size {batch_size}'
    if epochs is not None:
        args_str += f' --epochs {epochs}'
    if ft_epochs_udf_only is not None:
        args_str += f' --ft_epochs_udf_only {ft_epochs_udf_only}'
    if early_stopping_patience is not None:
        args_str += f' --early_stopping_patience {early_stopping_patience}'
    if test_against is not None:
        args_str += f' --test_against {test_against}'
    if train_on_test is not None:
        args_str += f' --train_on_test {train_on_test}'
    if stratify_dataset_by_runtimes is not None:
        args_str += f' --stratify_dataset_by_runtimes {stratify_dataset_by_runtimes}'
    if stratify_per_database_by_runtimes is not None:
        args_str += f' --stratify_per_database_by_runtimes {stratify_per_database_by_runtimes}'
    if stratification_prioritize_loops is not None:
        args_str += f' --stratification_prioritize_loops {stratification_prioritize_loops}'
    if skip_udf is not None:
        args_str += f' --skip_udf {skip_udf}'
    if mp_ignore_udf:
        args_str += f' --mp_ignore_udf {mp_ignore_udf}'
    if quick_train:
        args_str += f' --max_epoch_tuples 700'
    if zs_paper_dataset:
        args_str += f' --zs_paper_dataset'
    if plans_have_no_udf:
        args_str += f' --plans_have_no_udf'
    if pretrained_model_artifact_dir is not None:
        args_str += f' --pretrained_model_artifact_dir {pretrained_model_artifact_dir}'
    if pretrained_model_filename is not None:
        args_str += f' --pretrained_model_filename {pretrained_model_filename}'
    if train_udf_graph_against_udf_runtime:
        args_str += f' --train_udf_graph_against_udf_runtime'
    if work_with_udf_repr:
        args_str += f' --work_with_udf_repr'
    if max_runtime is not None:
        args_str += f' --max_runtime {max_runtime}'
    if include_no_udf_data:
        args_str += f' --include_no_udf_data'
    if include_pullup_data:
        args_str += f' --include_pullup_data'
    if include_pushdown_data:
        args_str += f' --include_pushdown_data'
    if include_no_udf_data_large:
        args_str += f' --include_no_udf_data_large'
    if include_select_only_w_branch:
        args_str += f' --include_select_only_w_branch'
    if min_runtime_ms is not None:
        args_str += f' --min_runtime_ms {min_runtime_ms}'
    if card_type is not None:
        args_str += f' --card_type {card_type}'

    if test_with_count_edges_msg_aggr:
        args_str += f' --test_with_count_edges_msg_aggr'

    if min_num_branches is not None:
        args_str += f' --min_num_branches {min_num_branches}'
    if max_num_branches is not None:
        args_str += f' --max_num_branches {max_num_branches}'
    if min_num_loops is not None:
        args_str += f' --min_num_loops {min_num_loops}'
    if max_num_loops is not None:
        args_str += f' --max_num_loops {max_num_loops}'
    if min_num_np_calls is not None:
        args_str += f' --min_num_np_calls {min_num_np_calls}'
    if max_num_np_calls is not None:
        args_str += f' --max_num_np_calls {max_num_np_calls}'
    if min_num_math_calls is not None:
        args_str += f' --min_num_math_calls {min_num_math_calls}'
    if max_num_math_calls is not None:
        args_str += f' --max_num_math_calls {max_num_math_calls}'
    if min_num_comp_nodes is not None:
        args_str += f' --min_num_comp_nodes {min_num_comp_nodes}'
    if max_num_comp_nodes is not None:
        args_str += f' --max_num_comp_nodes {max_num_comp_nodes}'

    cmd = f'python3 train.py --data_keyword {data_keyword} --model_config {model_config} --wl_base_path {wl_base_path} --out_base_path {out_base_path} --device {device} {args_str}'
    print(cmd)
    return cmd


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts_path', type=str)
    parser.add_argument('--wl_base_path', type=str)

    args = parser.parse_args()
    artifacts_path = args.artifacts_path
    wl_base_path = args.wl_base_path

    register_at_wandb = False
    wandb_entity = '<YOUR_WANDB_ENTITY>'
    wandb_project = '<YOUR_WANDB_PROJECT>'

    for dataset in dataset_list_dict['zs_less_scaled']:
        # train model
        gen_cmd(data_keyword='complex_dd', model_config='ddactfonudf_liboh_gradnorm_mldupl',
                device=f'cuda:0', test_against=dataset.db_name, stratify_per_database_by_runtimes=True,
                min_runtime_ms=50, epochs=60, ft_epochs_udf_only=0, include_pullup_data=True, include_no_udf_data=True,
                include_no_udf_data_large=False, card_type='act', apply_pca_evaluation=False, num_workers=24,
                artifacts_path=artifacts_path, wl_base_path=wl_base_path, register_at_wandb=register_at_wandb,
                wandb_entity=wandb_entity, wandb_project=wandb_project)

    # ablation study
    gen_cmd(data_keyword='complex_dd', model_config='ddactfonudf_liboh_gradnorm_mldupl_loopend_loopedge',
            test_against='genome', stratify_per_database_by_runtimes=True,
            min_runtime_ms=50, epochs=60, ft_epochs_udf_only=0, include_pullup_data=True, include_no_udf_data=True,
            card_type='act', apply_pca_evaluation=True, artifacts_path=artifacts_path, wl_base_path=wl_base_path,
            register_at_wandb=register_at_wandb, wandb_entity=wandb_entity, wandb_project=wandb_project),
    gen_cmd(data_keyword='complex_dd', model_config='ddactfonudf_liboh_gradnorm_mldupl_loopend',
            test_against='genome', stratify_per_database_by_runtimes=True,
            min_runtime_ms=50, epochs=60, ft_epochs_udf_only=0, include_pullup_data=True, include_no_udf_data=True,
            card_type='act', apply_pca_evaluation=True, artifacts_path=artifacts_path, wl_base_path=wl_base_path,
            register_at_wandb=register_at_wandb, wandb_entity=wandb_entity, wandb_project=wandb_project),
    gen_cmd(data_keyword='complex_dd', model_config='ddactfonudf_liboh_gradnorm_mldupl',
            test_against='genome', stratify_per_database_by_runtimes=True,
            min_runtime_ms=50, epochs=60, ft_epochs_udf_only=0, include_pullup_data=True, include_no_udf_data=True,
            card_type='act', apply_pca_evaluation=True, artifacts_path=artifacts_path, wl_base_path=wl_base_path,
            register_at_wandb=register_at_wandb, wandb_entity=wandb_entity, wandb_project=wandb_project),
    gen_cmd(data_keyword='complex_dd', model_config='ddactfonudf_liboh_mldupl',
            test_against='genome', stratify_per_database_by_runtimes=True,
            min_runtime_ms=50, epochs=60, ft_epochs_udf_only=0, include_pullup_data=True, include_no_udf_data=True,
            card_type='act', apply_pca_evaluation=True, artifacts_path=artifacts_path, wl_base_path=wl_base_path,
            register_at_wandb=register_at_wandb, wandb_entity=wandb_entity, wandb_project=wandb_project),
    gen_cmd(data_keyword='complex_dd', model_config='ddact_liboh_mldupl',
            test_against='genome', stratify_per_database_by_runtimes=True,
            min_runtime_ms=50, epochs=60, ft_epochs_udf_only=0, include_pullup_data=True, include_no_udf_data=True,
            card_type='act', apply_pca_evaluation=True, artifacts_path=artifacts_path, wl_base_path=wl_base_path,
            register_at_wandb=register_at_wandb, wandb_entity=wandb_entity, wandb_project=wandb_project),
