import argparse
import os.path

from prepare_db.augment_udf_specific_stats import create_db_statistics
from prepare_db.generate_column_stats import generate_stats
from prepare_db.generate_string_statistics import generate_string_stats
from run_benchmark import StoreDictKeyPair

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, required=True)
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--dbs', required=True, type=str)
    parser.add_argument('--dbms', default='postgres')  # or duckdb
    parser.add_argument('--dbms_kwargs', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")

    args = parser.parse_args()

    assert os.path.exists(args.work_dir), f"Work directory {args.work_dir} does not exist"

    # generate stats
    print('Generate col stats')
    generate_stats(args.data_dir, os.path.join(args.work_dir, 'statistics'), datasets=args.dbs, force=False)
    print('Generate string stats')
    generate_string_stats(args.data_dir, os.path.join(args.work_dir, 'statistics'), datasets=args.dbs, force=False)

    print('Extract udf specific dataset stats')
    create_db_statistics(dbms=args.dbms, dbms_kwargs=args.dbms_kwargs,
                         col_stats_dir=os.path.join(args.work_dir, 'statistics'),
                         target=os.path.join(args.work_dir, 'db_metadata.csv'), datasets=args.dbs)

    print("Finished generating stats")
