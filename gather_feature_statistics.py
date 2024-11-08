import argparse
import glob
import os

from models.preprocessing.feature_statistics import gather_feature_statistics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # gather statistics
    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--workload_runs', default=None, nargs='+')
    parser.add_argument('--no_udfs', action='store_true')

    args = parser.parse_args()

    # gather_feature_statistics
    workload_runs = []

    for wl in args.workload_runs:
        workload_runs += glob.glob(f'{args.raw_dir}/*/{wl}')

    gather_feature_statistics(workload_runs, os.path.join(args.raw_dir, 'statistics_workload_combined.json'),
                              os.path.join(args.raw_dir, 'udf_stats.json') if not args.no_udfs else None)

    print('Done')
