import argparse
import functools
import multiprocessing
import os

from cross_db_benchmark.datasets.datasets import dataset_list_dict
from prepare_db.clean_DBs import clean_up_DBs
from run_benchmark import StoreDictKeyPair

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbms', default='duckdb')
    parser.add_argument('--dbms_kwargs', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    # parser.add_argument('--meta_data_path', type=str)
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--dbs', default='zs_less_scaled', required=True, type=str)

    args = parser.parse_args()

    # check if the directory ./cleanDB_scripts exists and if not, then create it
    if not os.path.exists(os.path.join(args.work_dir, 'cleanDB_scripts')):
        os.makedirs(os.path.join(args.work_dir, 'cleanDB_scripts'))
    else:
        # delete all files in the directory
        files = os.listdir(os.path.join(args.work_dir, 'cleanDB_scripts'))
        for file in files:
            os.remove(os.path.join(args.work_dir, 'cleanDB_scripts', file))

    assert os.path.exists(os.path.join(args.work_dir,
                                       'db_metadata.csv')), f"Metadata file {os.path.join(args.work_dir, 'db_metadata.csv')} does not exist. Please run gen_dataset_stats.py before"

    # generate cleanup scripts and run cleanup
    fn = functools.partial(clean_up_DBs, dbms=args.dbms, dbms_kwargs=args.dbms_kwargs,
                           work_dir=args.work_dir)
    with multiprocessing.Pool() as pool:
        pool.map(fn, [db.db_name for db in dataset_list_dict[args.dbs]])

    print('Please regenerate the DB_metadata.csv file, hence num rows / ... might have been changed')

    print('Done!')
