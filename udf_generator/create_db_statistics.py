import argparse
import os

from UDF_graph.create_graph import StoreDictKeyPair
from data_prep.create_db_statistics import create_db_statistics

if __name__ == '__main__':
    # Parse the arguments for graph creation
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbms', default='postgres')  # or duckdb
    parser.add_argument('--dbms_kwargs', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument('--col_stats_dir', type=str)
    parser.add_argument('--target', type=str)

    args = parser.parse_args()

    os.makedirs(args.col_stats_dir, exist_ok=True)
    create_db_statistics(dbms=args.dbms, dbms_kwargs=args.dbms_kwargs, col_stats_dir=args.col_stats_dir,
                         target=args.target)

    print('Done')
