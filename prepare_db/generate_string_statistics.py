import collections
import functools
import json
import multiprocessing
import os
import traceback

import pandas as pd

from cross_db_benchmark.benchmark_tools.column_types import Datatype
from cross_db_benchmark.benchmark_tools.utils import load_schema_json, load_column_statistics, translate_col_names, \
    orig_col_names
from cross_db_benchmark.datasets.datasets import dataset_list_dict


def generate_string_stats(data_dir: str, out_path: str, datasets: str, force=True, max_sample_vals=100000,
                          min_str_occ=0.01,
                          verbose=False):
    fn = functools.partial(gen_string_stats_fn, data_dir=data_dir, out_path=out_path, force=force,
                           max_sample_vals=max_sample_vals, min_str_occ=min_str_occ, verbose=verbose)

    tasks = [(db.db_name, db.data_folder) for db in dataset_list_dict[datasets]]

    with multiprocessing.Pool() as pool:
        pool.map(fn, tasks)


def gen_string_stats_fn(args, data_dir: str, out_path: str, force: bool, max_sample_vals: int, min_str_occ: float,
                        verbose: bool):
    try:
        dataset, data_folder = args
        # read the schema file
        string_stats_path = os.path.join(out_path, f'{dataset}_string_statistics.json')
        if os.path.exists(string_stats_path) and not force:
            print(f"String stats already created ({dataset})")
            return

        schema = load_schema_json(dataset)
        column_stats = load_column_statistics(out_path, dataset)

        cols_with_freq_words = 0
        string_stats = dict()
        for table, cols in vars(column_stats).items():

            string_stats[table] = dict()
            if dataset == 'financial' and table == 'orders':
                table_dir = os.path.join(data_dir, data_folder, f'order.csv')
            else:
                table_dir = os.path.join(data_dir, data_folder, f'{table}.csv')
            assert os.path.exists(data_dir), f"Could not find table csv {table_dir}"
            if verbose:
                print(f"Generating string statistics for {table}")

            df_table = pd.read_csv(table_dir, nrows=max_sample_vals, **vars(schema.csv_kwargs))

            for c, col_stats in vars(cols).items():
                if col_stats.datatype in {str(Datatype.CATEGORICAL), str(Datatype.MISC)}:

                    # for the udfs we have adjusted some column names which otherwise would not be usable with python
                    # e.g. Month/Year -> Month_Year, 2B -> SecondB, 3B -> ThirdB
                    translated_c = translate_col_names(dataset, table, c)

                    col_vals = df_table[orig_col_names(dataset, table, c)]
                    # do not consider too many values
                    col_vals = col_vals[:max_sample_vals]
                    len_strs = len(col_vals)

                    # check how often a word occurs
                    word_vals = collections.defaultdict(int)
                    try:
                        split_col_vals = col_vals.str.split(' ')
                    except:
                        continue

                    for scol_vals in split_col_vals:
                        if not isinstance(scol_vals, list):
                            continue
                        for v in scol_vals:
                            if not isinstance(v, str):
                                continue
                            word_vals[v] += 1

                    # how often should a word appear
                    min_expected_occ = max(int(len_strs * min_str_occ), 1)

                    freq_str_words = list()
                    for val, occ in word_vals.items():
                        if occ > min_expected_occ:
                            freq_str_words.append(val)

                    if len(freq_str_words) > 0:
                        if verbose:
                            print(f"Found {len(freq_str_words)} frequent words for {c} "
                                  f"(expected {min_expected_occ}/{len_strs})")

                        cols_with_freq_words += 1
                        string_stats[table][translated_c] = dict(freq_str_words=freq_str_words)

        # save to json
        with open(string_stats_path, 'w') as outfile:
            print(f"Found {cols_with_freq_words} string-queryable columns for dataset {dataset}")
            # workaround for numpy and other custom datatypes
            json.dump(string_stats, outfile)
    except Exception as e:
        print(args, flush=True)
        traceback.print_exc()
        raise e
