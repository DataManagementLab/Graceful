import argparse
import multiprocessing
import multiprocessing as mp
import os
import time

from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.generate_workload import generate_workload
from cross_db_benchmark.benchmark_tools.parse_run import parse_run
from cross_db_benchmark.datasets.datasets import dataset_list_dict
from cross_db_benchmark.meta_tools.download_relational_fit import download_from_relational_fit
from run_benchmark import StoreDictKeyPair

workload_defs = {
    # standard workloads will be capped at 5k
    # 'workload_100k_s1': dict(num_queries=100000,
    #                          max_no_predicates=5,
    #                          max_no_aggregates=3,
    #                          max_no_group_by=0,
    #                          max_cols_per_agg=2,
    #                          seed=1),
    # for complex predicates, this will be capped at 5k
    # 'complex_workload_200k_s1': dict(num_queries=200000,
    #                                  max_no_predicates=5,
    #                                  max_no_aggregates=3,
    #                                  max_no_group_by=0,
    #                                  max_cols_per_agg=2,
    #                                  complex_predicates=True,
    #                                  seed=1),
    # # for index workloads, will also be capped at 5k
    # 'workload_100k_s2': dict(num_queries=100000,
    #                          max_no_predicates=5,
    #                          max_no_aggregates=3,
    #                          max_no_group_by=0,
    #                          max_cols_per_agg=2,
    #                          seed=2),
    'workload_100k_s1_udf': dict(num_queries=100000,
                                 max_no_predicates=5,
                                 max_no_aggregates=1,
                                 # with more aggregates the udf processing failes: assumes when UDF is in aggr, there is only 1 aggregation - idk why (probably a hacky impl)
                                 max_no_group_by=0,
                                 max_cols_per_agg=2,
                                 seed=1),
}


def compute(input):
    source, target, d, wl, parse_baseline, cap_queries, udf_code_location = input
    no_plans, stats = parse_run(source, target, d, args.database, min_query_ms=args.min_query_ms,
                                cap_queries=cap_queries,
                                parse_baseline=parse_baseline, udf_code_location=udf_code_location,
                                parse_join_conds=True)
    return dict(dataset=d, workload=wl, no_plans=no_plans, **stats)


def workload_gen(input):
    source_dataset, workload_path, max_no_joins, udf_stats_path, workload_args, col_stats_dir = input
    generate_workload(source_dataset, workload_path, col_stats_dir=col_stats_dir, max_no_joins=max_no_joins,
                      udf_stats_path=udf_stats_path,
                      **workload_args)
    print(f'Finished: {workload_path}')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--workload_dir', default='../zero-shot-data/workloads')
    parser.add_argument("--database_conn", dest='database_conn_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--database_kwargs", dest='database_kwarg_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument('--hardware', default='c8220')

    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--parsed_plan_dir', default=None)
    # parser.add_argument('--target_stats_path', default=None)
    parser.add_argument('--workloads', nargs='+', default=None)
    parser.add_argument('--min_query_ms', default=100, type=int)
    parser.add_argument('--cap_queries', default=5000, type=int)
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))
    parser.add_argument('--udf_code_location', default=None, type=str)

    parser.add_argument('--generate_column_statistics', action='store_true')
    parser.add_argument('--generate_string_statistics', action='store_true')
    parser.add_argument('--download_relational_fit', action='store_true')
    parser.add_argument('--scale_datasets', action='store_true')
    parser.add_argument('--load_database', action='store_true')
    parser.add_argument('--generate_workloads', action='store_true')
    parser.add_argument('--print_run_commands', action='store_true')
    parser.add_argument('--parse_all_queries', action='store_true')
    parser.add_argument('--print_zero_shot_commands', action='store_true')

    parser.add_argument('--out_stats_dir', default=None)
    parser.add_argument('--col_stats_dir', default=None)
    parser.add_argument('--dbs', default=None, type=str)

    args = parser.parse_args()

    if args.database_kwarg_dict is None:
        args.database_kwarg_dict = dict()

    if args.download_relational_fit:

        print("Downloading datasets from relational.fit...")
        for rel_fit_dataset_name, dataset_name in tqdm([('Walmart', 'walmart'),
                                                        ('Basketball_men', 'basketball'),
                                                        ('financial', 'financial'),
                                                        ('geneea', 'geneea'),
                                                        ('Accidents', 'accidents'),
                                                        ('imdb_MovieLens', 'movielens'),
                                                        ('lahman_2014', 'baseball'),
                                                        ('Hepatitis_std', 'hepatitis'),
                                                        ('NCAA', 'tournament'),
                                                        ('VisualGenome', 'genome'),
                                                        ('Credit', 'credit'),
                                                        ('employee', 'employee'),
                                                        ('Carcinogenesis', 'carcinogenesis'),
                                                        ('ConsumerExpenditures', 'consumer'),
                                                        ('Seznam', 'seznam'),
                                                        ('FNHK', 'fhnk')]):
            download_from_relational_fit(rel_fit_dataset_name, dataset_name, root_data_dir=args.data_dir)

    # if args.scale_datasets:
    #     # scale if required
    #     for dataset in database_list:
    #         if dataset.scale == 1 and not dataset.down_scale:
    #             continue
    #         elif dataset.down_scale:
    #             assert dataset.data_folder != dataset.source_dataset, "For scaling a new folder is required"
    #             print(f"Scaling dataset {dataset.db_name}")
    #             curr_source_dir = os.path.join(args.data_dir, dataset.source_dataset)
    #             curr_target_dir = os.path.join(args.data_dir, dataset.data_folder)
    #             if not os.path.exists(curr_target_dir):
    #                 scale_down_dataset(dataset.source_dataset, curr_source_dir, curr_target_dir)
    #         else:
    #             assert dataset.data_folder != dataset.source_dataset, "For scaling a new folder is required"
    #             print(f"Scaling dataset {dataset.db_name}")
    #             curr_source_dir = os.path.join(args.data_dir, dataset.source_dataset)
    #             curr_target_dir = os.path.join(args.data_dir, dataset.data_folder)
    #             if not os.path.exists(curr_target_dir):
    #                 scale_up_dataset(dataset.source_dataset, curr_source_dir, curr_target_dir, scale=dataset.scale,
    #                                  scale_individually=dataset.scale_individually)
    #
    # if args.load_database:
    #     # load databases
    #     # also load imdb full dataset to be able to run the full job benchmark
    #     for dataset in ext_database_list:
    #         for database in [DatabaseSystem.POSTGRES]:
    #             curr_data_dir = os.path.join(args.data_dir, dataset.data_folder)
    #             print(f"Loading database {dataset.db_name} from {curr_data_dir}")
    #             load_database(curr_data_dir, dataset.source_dataset, database, dataset.db_name, args.database_conn_dict,
    #                           args.database_kwarg_dict)

    if args.generate_workloads:
        workload_gen_setups = []
        for dataset in dataset_list_dict['zs']:
            for workload_name, workload_args in workload_defs.items():
                print(f'{dataset} - {workload_args}')
                workload_path = os.path.join(args.workload_dir, dataset.db_name, f'{workload_name}.sql')
                udf_stats_path = os.path.join(args.workload_dir, dataset.db_name, f'{workload_name}_udf_stats.json')
                workload_gen_setups.append(
                    (dataset.source_dataset, workload_path, dataset.max_no_joins, udf_stats_path, workload_args,
                     args.col_stats_dir))

        start_t = time.perf_counter()
        proc = multiprocessing.cpu_count() - 2
        p = mp.Pool(initargs=('arg',), processes=proc)
        p.map(workload_gen, workload_gen_setups)
        print(f"Generated workloads in {time.perf_counter() - start_t:.2f} secs")

    print("Done")
