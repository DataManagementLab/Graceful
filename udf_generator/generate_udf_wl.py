import argparse
import os

from data_prep.generate_udf_wl import setup_exp_folder, dataset_generator
from datasets import datasets_list

if __name__ == '__main__':
    # Parse the arguments for graph creation
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbms', default='postgres')  # or duckdb
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--no_funcs', type=int, default=None)
    parser.add_argument('--wl_udf_stats', default=None)  # input is something like "udf_stats.json"
    parser.add_argument('--setup', default=False, type=bool)  # True or False
    parser.add_argument('--exact_tree', action='store_true')
    parser.add_argument('--experiment_dir', type=str)
    parser.add_argument('--db_metadata_path', type=str)
    args = parser.parse_args()

    if args.setup:
        setup_exp_folder(database_lst=datasets_list, folder_loc=args.experiment_dir, exp_folder_name=args.exp_name)
    else:
        if args.wl_udf_stats is None:  # create UDFs for select-only queries
            dataset_generator(database_list=datasets_list,
                              exp_folder_path=os.path.join(args.experiment_dir, args.exp_name, "dbs"),
                              queries_per_db=args.no_funcs, with_branch=True,
                              exact_tree=args.exact_tree, meta_file=False, dbms=args.dbms,
                              db_metadata_path=args.db_metadata_path)
        else:  # create UDFs for SPAJ queries that were already generated before with the other repository
            dataset_generator(database_list=datasets_list,
                              exp_folder_path=os.path.join(args.experiment_dir, args.exp_name, "dbs"),
                              queries_per_db=args.no_funcs, with_branch=True, exact_tree=args.exact_tree,
                              meta_file=args.wl_udf_stats, dbms=args.dbms, db_metadata_path=args.db_metadata_path)

    print('Done.')
