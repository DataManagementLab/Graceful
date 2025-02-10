import argparse
import os

from cross_db_benchmark.datasets.datasets import zs_dataset_less_scaled_list
from udf_graph.create_graph import prepareGraphs, StoreDictKeyPair
from udf_graph.dbms_wrapper import DBMSWrapper
from udf_graph.gather_feature_stats import gather_feature_stats_wrapper, FEATURE_DICT

if __name__ == '__main__':
    # Parse the arguments for graph creation
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--gather', default=None)
    parser.add_argument('--dbms', default='duckdb')  # or duckdb
    parser.add_argument('--dbms_kwargs', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument('--add_loop_end_node', default=False, action='store_true')
    parser.add_argument('--card_est_assume_lazy_eval', default=False, action='store_true')

    parser.add_argument('--dataset_list_name', default='zs_less_scaled')
    parser.add_argument('--duckdb_dir', default=None, type=str)
    parser.add_argument('--deepdb_dir', default=None, type=str)

    parser.add_argument('--skip_wj', default=False, action='store_true')
    args = parser.parse_args()

    exp_folder = args.exp_folder
    dataset = args.dataset

    if args.gather is None:  # determine if we build graph or gather the feature stats
        graph_kwargs = {'add_loop_end_node': args.add_loop_end_node}

        if 'intermed' in exp_folder.lower():
            udf_intermed_pos=True
            print(f'Assume intermed udfs in exp folder {exp_folder}')
            pullup_udf = False
        elif 'pullup' in exp_folder.lower() or 'pull_up' in exp_folder.lower():
            pullup_udf = True
            print(f'Assume pullup udfs in exp folder {exp_folder}')
            udf_intermed_pos = False
        else:
            pullup_udf = False
            udf_intermed_pos = False


        duckdb_kwargs = {
            'database': os.path.join(args.duckdb_dir, f'{args.dataset}_10_1.db'),
            # 'read_only': True,
        }

        db = [d for d in zs_dataset_less_scaled_list if d.db_name == dataset][0]

        dbms_wrapper = DBMSWrapper(dbms=args.dbms, dbms_kwargs=args.dbms_kwargs, db_name=args.dataset)
        prepareGraphs(code_location=os.path.join(args.exp_folder, "dbs", args.dataset, "sql_scripts"),
                      graph_location=os.path.join(args.exp_folder, "dbs", args.dataset, "created_graphs"),
                      pullup_udf=pullup_udf, udf_intermed_pos=udf_intermed_pos, exp_folder=args.exp_folder,
                      func_tab_map=os.path.join(args.exp_folder, "dbs", args.dataset, "func_table_dict.csv"),
                      db_name=args.dataset, dbms_wrapper=dbms_wrapper, graph_kwargs=graph_kwargs,
                      card_est_assume_lazy_eval=args.card_est_assume_lazy_eval,
                      duckdb_kwargs=duckdb_kwargs, skip_wj=args.skip_wj,
                      deepdb_rel_ensemble_location=os.path.join(args.deepdb_dir, db.data_folder,
                                                                f'spn_ensembles/ensemble_relationships_{db.data_folder}_0.3_10000000.pkl'),
                      deepdb_single_ensemble_location=os.path.join(args.deepdb_dir, db.data_folder,
                                                                   f'spn_ensembles/ensemble_single_{db.data_folder}_0.3_10000000.pkl'))
    else:
        gather_feature_stats_wrapper(args.exp_folder, FEATURE_DICT, dataset_list_name=args.dataset_list_name)

    print('Done')
