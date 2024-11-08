# Add the parent directory to the system path
import json
import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import multiprocessing
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
from tqdm import tqdm

from deepdb.data_preparation.join_data_preparation import prepare_sample_hdf
from deepdb.data_preparation.prepare_single_tables import prepare_all_tables
from deepdb.datasets.datasets_lists import dataset_list_dict
from deepdb.datasets.generate_schema import gen_schema
from deepdb.datasets.utils import load_json
from deepdb.ensemble_compilation.spn_ensemble import read_ensemble
from deepdb.ensemble_creation.naive import create_naive_all_split_ensemble, naive_every_relationship_ensemble
from deepdb.ensemble_creation.rdc_based import candidate_evaluation
from deepdb.evaluation.augment_cards import augment_cardinalities
from deepdb.evaluation.utils import save_csv
from deepdb.rspn.code_generation.generate_code import generate_ensemble_code

np.random.seed(1)


def gen_hdf(schema, max_rows_per_hdf_file, hdf_path, csv_path):
    logger.info(f"Generating HDF files for tables in {csv_path} and store to path {hdf_path}")
    if os.path.exists(hdf_path):
        logger.info(f"Removing target path {hdf_path}")
        shutil.rmtree(hdf_path)
    logger.info(f"Making target path {hdf_path}")
    os.makedirs(hdf_path)
    prepare_all_tables(schema, hdf_path, max_table_data=max_rows_per_hdf_file)
    logger.info(f"Files successfully created")


# def compute(input):
#     data_folder, db_name, scale, wl_name = input
#
#     csv_path = os.path.join(args.csv_path, data_folder)
#
#     # find correct SPN ensemble
#     ensemble_dir = os.path.join(args.ensemble_location[0], db_name, 'spn_ensembles')
#     ensemble_locations = []
#     for ens_name in [f'ensemble_relationships_{db_name}_0.3_10000000.pkl',
#                      f'ensemble_single_{db_name}_0.3_10000000.pkl']:
#         ensemble_loc = os.path.join(ensemble_dir, ens_name)
#         if os.path.exists(ensemble_loc):
#             ensemble_locations.append(ensemble_loc)
#     if len(ensemble_locations) == 1:
#         ensemble_loc = os.path.join(ensemble_dir, f'ensemble_relationships_{db_name}_0.5_10000000.pkl')
#         if os.path.exists(ensemble_loc):
#             ensemble_locations.append(ensemble_loc)
#
#     if len(ensemble_locations) == 0:
#         ensemble_locations = None
#         print(f"No ensemble found for {db_name}")
#     print(f"Using ensemble {ensemble_locations}")
#
#     # find workloads
#     source_dir = os.path.join(args.source_workload_dir, db_name)
#     target_dir = os.path.join(args.workload_target_paths[0], db_name)
#
#     schema = gen_schema(db_name, csv_path)
#
#     # loop over workloads
#     src_p = os.path.join(source_dir, wl_name)
#     dest_p = os.path.join(target_dir, wl_name)
#
#     if not os.path.exists(src_p):
#         print(f"Path does not exist: {src_p}")
#         return
#
#     if args.overwrite_scale is not None:
#         print(f"Overwriting scale to {args.overwrite_scale}")
#         scale = args.overwrite_scale
#
#     if args.scaling_dir is not None:
#         scaling_json = load_json(os.path.join(args.scaling_dir, db_name, args.scaling_filename),
#                                  namespace=True)
#         scale = scaling_json.no_replications + 1
#         scale = 2 ** scale
#         print(f"Read scale {scale}")
#
#     augment_cardinalities(schema, ensemble_locations, src_p, dest_p,
#                           rdc_spn_selection=args.rdc_spn_selection,
#                           pairwise_rdc_path=args.pairwise_rdc_path,
#                           merge_indicator_exp=args.merge_indicator_exp,
#                           exploit_overlapping=args.exploit_overlapping, max_variants=args.max_variants,
#                           scale=scale)

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split("&&"):
            k, v = kv.split("=")

            if v == 'True':
                v = True
            elif v == 'False':
                v = False
            elif v.startswith('[') and v.endswith(']'):
                v = v[1:-1].split(',')
                v = [e[1:-1] if e.startswith('\'') and e.endswith('\'') else e for e in v]

            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ssb-500gb', help='Which dataset to be used')

    # generate hdf
    parser.add_argument('--generate_hdf', help='Prepare hdf5 files for single tables', action='store_true')
    parser.add_argument('--generate_all_hdfs', help='Prepare hdf5 files for single tables', action='store_true')
    parser.add_argument('--train_all_models', help='Train the DeepDB models', action='store_true')
    parser.add_argument('--generate_sampled_hdfs', help='Prepare hdf5 files for single tables', action='store_true')
    parser.add_argument('--csv_seperator', default='|')
    parser.add_argument('--csv_path', default=None)
    parser.add_argument('--hdf_path', default='../ssb-benchmark/gen_hdf')
    parser.add_argument('--max_rows_per_hdf_file', type=int, default=20000000)
    parser.add_argument('--hdf_sample_size', type=int, default=1000000)

    # generate ensembles
    parser.add_argument('--generate_ensemble', help='Trains SPNs on schema', action='store_true')
    parser.add_argument('--ensemble_strategy', default='single')
    parser.add_argument('--ensemble_path', default='../ssb-benchmark/spn_ensembles')
    parser.add_argument('--pairwise_rdc_path', default=None)
    parser.add_argument('--samples_rdc_ensemble_tests', type=int, default=10000)
    parser.add_argument('--samples_per_spn', help="How many samples to use for joins with n tables",
                        nargs='+', type=int, default=[10000000, 10000000, 2000000, 2000000])
    parser.add_argument('--post_sampling_factor', nargs='+', type=int, default=[30, 30, 2, 1])
    parser.add_argument('--rdc_threshold', help='If RDC value is smaller independence is assumed', type=float,
                        default=0.5)
    parser.add_argument('--bloom_filters', help='Generates Bloom filters for grouping', action='store_true')
    parser.add_argument('--ensemble_budget_factor', type=int, default=5)
    parser.add_argument('--ensemble_max_no_joins', type=int, default=3)
    parser.add_argument('--incremental_learning_rate', type=int, default=0)
    parser.add_argument('--incremental_condition', type=str, default=None)
    parser.add_argument('--min_min_instance_slice', type=int, default=500)

    # generate code
    parser.add_argument('--code_generation', help='Generates code for trained SPNs for faster Inference',
                        action='store_true')
    parser.add_argument('--use_generated_code', action='store_true')

    # ground truth
    parser.add_argument('--aqp_ground_truth', help='Computes ground truth for AQP', action='store_true')
    parser.add_argument('--cardinalities_ground_truth', help='Computes ground truth for Cardinalities',
                        action='store_true')

    # evaluation
    parser.add_argument('--source_workload_dir', default=None)
    parser.add_argument('--augment_cardinalities', help='Evaluates SPN ensemble to compute cardinalities',
                        action='store_true')
    parser.add_argument('--gen_ensemble_stats', help='Generates statistics about SPN ensembles',
                        action='store_true')
    parser.add_argument('--augment_all_workload_cardinalities', help='Evaluates SPN ensemble to compute cardinalities',
                        action='store_true')
    parser.add_argument('--rdc_spn_selection', help='Uses pairwise rdc values to for the SPN compilation',
                        action='store_true')
    parser.add_argument('--ensemble_location', nargs='+',
                        default=['../ssb-benchmark/spn_ensembles/ensemble_single_ssb-500gb_10000000.pkl',
                                 '../ssb-benchmark/spn_ensembles/ensemble_relationships_ssb-500gb_10000000.pkl'])
    parser.add_argument('--query_file_location', default='./benchmarks/ssb/sql/cardinality_queries.sql')
    parser.add_argument('--ground_truth_file_location',
                        default='./benchmarks/ssb/sql/cardinality_true_cardinalities_100GB.csv')
    parser.add_argument('--database_name', default=None)
    parser.add_argument('--target_path', default='../ssb-benchmark/results')
    parser.add_argument('--raw_folder', default='../ssb-benchmark/results')
    parser.add_argument('--confidence_intervals', help='Compute confidence intervals', action='store_true')
    parser.add_argument('--max_variants', help='How many spn compilations should be computed for the cardinality '
                                               'estimation. Seeting this parameter to 1 means greedy strategy.',
                        type=int, default=1)
    parser.add_argument('--no_exploit_overlapping', action='store_true')
    parser.add_argument('--no_merge_indicator_exp', action='store_true')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--overwrite_scale', type=int, default=None)
    parser.add_argument('--scaling_dir', type=str, default=None)
    parser.add_argument('--scaling_filename', type=str, default=None)

    # evaluation of spn ensembles in folder
    parser.add_argument('--hdf_build_path', default='')

    # log level
    parser.add_argument('--log_level', type=int, default=logging.DEBUG)

    parser.add_argument('--source_workloads', default=None, nargs='+')
    parser.add_argument('--workload_target_paths', default=None, nargs='+')

    parser.add_argument('--datasets_set', default=None)

    # dbms
    parser.add_argument('--dbms', default='duckdb')
    parser.add_argument('--duckdb_dir', default=None)

    parser.add_argument('--csv_read_kwargs', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1&&KEY2=VAL2...")

    args = parser.parse_args()

    args.exploit_overlapping = not args.no_exploit_overlapping
    args.merge_indicator_exp = not args.no_merge_indicator_exp

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=args.log_level,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    if args.generate_all_hdfs:
        assert args.datasets_set is not None


        def gen_hdf_worker(task):
            db_name, csv_path, hdf_path = task
            try:

                if args.csv_read_kwargs is not None:
                    # convert string arg to dict using json
                    csv_read_kwargs = args.csv_read_kwargs
                else:
                    csv_read_kwargs = None

                # generate schema
                schema = gen_schema(db_name, csv_path, csv_read_kwargs=csv_read_kwargs)

                # run gen_hdf
                gen_hdf(schema, args.max_rows_per_hdf_file, hdf_path, csv_path)
            except Exception as e:
                print(e, flush=True)
                print(f'Error with: {db_name}', flush=True)
                traceback.print_exc()


        tasks = []
        for db in dataset_list_dict[args.datasets_set]:
            print(f"Generating HDF files for {db.db_name}")
            # intentionally do not use the scaled version
            hdf_path = os.path.join(args.hdf_path, db.data_folder, 'gen_single_light')

            if os.path.exists(hdf_path):
                print(f'Skip because already exists: {hdf_path}')
                continue

            csv_path = os.path.join(args.csv_path, db.data_folder)
            tasks.append((db.db_name, csv_path, hdf_path))

        parallelize = True
        if not parallelize:
            for task in tasks:
                gen_hdf_worker(task)
        else:
            with multiprocessing.Pool() as pool:
                pool.map(gen_hdf_worker, tasks)

    elif args.gen_ensemble_stats:
        spn_stats = []
        for db in tqdm(dataset_list_dict[args.datasets_set]):
            ensemble_dir = os.path.join(args.ensemble_location[0], db.db_name, 'spn_ensembles')
            for ens_style, ens_name in [('binary', f'ensemble_relationships_{db.db_name}_0.3_10000000.pkl'),
                                        ('single', f'ensemble_single_{db.db_name}_0.3_10000000.pkl')]:
                ensemble_loc = os.path.join(ensemble_dir, ens_name)
                if os.path.exists(ensemble_loc):
                    spn_ensemble = read_ensemble(ensemble_loc, build_reverse_dict=True)
                    for spn in spn_ensemble.spns:
                        spn_stats.append({
                            'database': db.db_name,
                            'learn_time': spn.learn_time,
                            'ens_style': ens_style,
                        })
        save_csv(spn_stats, 'experiments/data/statistics/spn_stats.csv')

    else:
        # Generate schema
        schema = gen_schema(args.dataset, args.csv_path)
        # Generate HDF files for simpler sampling
        if args.generate_hdf:
            gen_hdf(schema, args.max_rows_per_hdf_file, args.hdf_path, args.csv_path)

        # Generate sampled HDF files for fast join calculations
        if args.generate_sampled_hdfs:
            logger.info(f"Generating sampled HDF files for tables in {args.csv_path} and store to path {args.hdf_path}")
            prepare_sample_hdf(schema, args.hdf_path, args.max_rows_per_hdf_file, args.hdf_sample_size)
            logger.info(f"Files successfully created")

        # Generate ensemble for cardinality schemas
        if args.generate_ensemble:

            if not os.path.exists(args.ensemble_path):
                os.makedirs(args.ensemble_path)

            if args.ensemble_strategy == 'single':
                create_naive_all_split_ensemble(schema, args.hdf_path, args.samples_per_spn[0], args.ensemble_path,
                                                args.dataset, args.bloom_filters, args.rdc_threshold,
                                                args.max_rows_per_hdf_file, args.post_sampling_factor[0],
                                                incremental_learning_rate=args.incremental_learning_rate)
            elif args.ensemble_strategy == 'relationship':
                naive_every_relationship_ensemble(schema, args.hdf_path, args.samples_per_spn[1], args.ensemble_path,
                                                  args.dataset, args.bloom_filters, args.rdc_threshold,
                                                  args.max_rows_per_hdf_file, args.post_sampling_factor[0],
                                                  incremental_learning_rate=args.incremental_learning_rate,
                                                  min_min_instance_slice=args.min_min_instance_slice)
            elif args.ensemble_strategy == 'rdc_based':
                logging.info(
                    f"maqp(generate_ensemble: ensemble_strategy={args.ensemble_strategy}, incremental_learning_rate={args.incremental_learning_rate}, incremental_condition={args.incremental_condition}, ensemble_path={args.ensemble_path})")
                candidate_evaluation(schema, args.hdf_path, args.samples_rdc_ensemble_tests, args.samples_per_spn,
                                     args.max_rows_per_hdf_file, args.ensemble_path, args.database_name,
                                     args.post_sampling_factor, args.ensemble_budget_factor, args.ensemble_max_no_joins,
                                     args.rdc_threshold, args.pairwise_rdc_path,
                                     incremental_learning_rate=args.incremental_learning_rate,
                                     incremental_condition=args.incremental_condition,
                                     db_kwargs={'duckdb_dir': args.duckdb_dir, 'dbms': args.dbms})
            else:
                raise NotImplementedError

        # Read pre-trained ensemble and evaluate cardinality queries scale
        if args.code_generation:
            spn_ensemble = read_ensemble(args.ensemble_path, build_reverse_dict=True)
            generate_ensemble_code(spn_ensemble, floating_data_type='float', ensemble_path=args.ensemble_path)

    print('Done.')
