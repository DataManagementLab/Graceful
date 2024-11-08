import argparse
import multiprocessing
import multiprocessing as mp
import os
import time

import pandas as pd

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.parse_run import parse_run
from cross_db_benchmark.datasets.datasets import dataset_list_dict


def compute(input):
    source, target, d, deepdb_rel_ensemble_location, deepdb_single_ensemble_location, deepdb_dataset_scale_factor, wl, parse_baseline, cap_queries, udf_code_location, duckdb_kwargs, skip_wj, skip_deepdb, keep_existing, prune_plans = input
    no_plans, stats = parse_run(source, target, d, [deepdb_single_ensemble_location, deepdb_rel_ensemble_location],
                                deepdb_dataset_scale_factor, args.database,
                                duckdb_kwargs=duckdb_kwargs,
                                pg_kwargs=None,
                                min_query_ms=args.min_query_ms,
                                cap_queries=cap_queries,
                                parse_baseline=parse_baseline, udf_code_location=udf_code_location,
                                parse_join_conds=True, skip_dump=args.skip_dump, skip_deepdb=skip_deepdb,
                                skip_wj=skip_wj, keep_existing=keep_existing, prune_plans=prune_plans)
    return dict(dataset=d, workload=wl, no_plans=no_plans, **stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--parsed_plan_dir', default=None)
    parser.add_argument('--workloads', nargs='+', default=None)
    parser.add_argument('--out_names', nargs='+', default=None)
    parser.add_argument('--min_query_ms', default=100, type=int)
    parser.add_argument('--cap_queries', default=5000, type=int)
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))
    parser.add_argument('--udf_code_location', default=None, type=str)
    parser.add_argument('--skip_dump', action='store_true')
    parser.add_argument('--dataset_list', default='zs_less_scaled', type=str)
    parser.add_argument('--duckdb_dir', default=None, type=str)

    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--skip_wj', action='store_true')
    parser.add_argument('--skip_deepdb', action='store_true')
    parser.add_argument('--keep_existing', action='store_true')
    parser.add_argument('--prune_plans', action='store_true')
    parser.add_argument('--deepdb_artifacts_dir', default=None, type=str)

    args = parser.parse_args()

    cap_queries = args.cap_queries
    if cap_queries == 'None':
        cap_queries = None

    udf_code_location = args.udf_code_location

    if args.out_names is None:
        out_names = args.workloads
    else:
        out_names = args.out_names

    setups = []
    for wl, out_name in zip(args.workloads, out_names):
        for db in [d for d in dataset_list_dict[args.dataset_list]]:
            d = db.db_name

            if args.dataset is not None:
                if d != args.dataset:
                    continue

            rel_ensemble_location = f'{args.deepdb_artifacts_dir}/{db.data_folder}/spn_ensembles/ensemble_relationships_{db.data_folder}_0.3_10000000.pkl'
            single_ensemble_location = f'{args.deepdb_artifacts_dir}/{db.data_folder}/spn_ensembles/ensemble_single_{db.data_folder}_0.3_10000000.pkl'
            scale = 1

            source = os.path.join(args.raw_dir, d, wl)
            parse_baseline = not 'complex' in wl
            if not os.path.exists(source):
                print(f"Missing: {d}: {wl}\n{source}")
                continue

            duckdb_kwargs = {
                'database': os.path.join(args.duckdb_dir, f'{d}_10_1.db'),
                'read_only': True,
            }

            target = os.path.join(args.parsed_plan_dir, d, out_name)

            # if os.path.exists(target):
            #     print(f"Skipping: {d} because already exists: {wl}")
            #     continue

            setups.append(
                (source, target, d, rel_ensemble_location, single_ensemble_location, scale, wl, parse_baseline,
                 cap_queries, udf_code_location,
                 duckdb_kwargs, args.skip_wj, args.skip_deepdb, args.keep_existing, args.prune_plans))

    start_t = time.perf_counter()

    if len(setups) == 0:
        print("No setups found")
        exit(0)

    parallelize = False
    if parallelize:
        proc = multiprocessing.cpu_count() - 2
        p = mp.Pool(initargs=('arg',), processes=proc)
        eval = p.map(compute, setups)
    else:
        eval = []
        for s in setups:
            print(f'Parse db: {s[2]} wl: {s[3]}')
            eval.append(compute(s))

    eval = pd.DataFrame(eval)
    print()
    print(eval[['dataset', 'workload', 'no_plans']].to_string(index=False))

    print()
    print(eval[['workload', 'no_plans']].groupby('workload').sum().to_string())

    print()
    print(f"Total plans parsed in {time.perf_counter() - start_t:.2f} secs: {eval.no_plans.sum()}")

    print("Done")
