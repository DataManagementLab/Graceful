import functools
import json
import multiprocessing
import os
import traceback

import numpy as np
import pandas as pd

from cross_db_benchmark.benchmark_tools.column_types import Datatype
from cross_db_benchmark.benchmark_tools.utils import load_schema_json, translate_col_names
from cross_db_benchmark.datasets.datasets import dataset_list_dict


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Datatype):
            return str(obj)
        else:
            return super(CustomEncoder, self).default(obj)


def column_stats(column, categorical_threshold=10000):
    """
    Default method for encoding the datasets
    """
    nan_ratio = sum(column.isna()) / len(column)
    stats = dict(nan_ratio=nan_ratio)
    if column.dtype == object:
        if len(column.unique()) > categorical_threshold:
            stats.update(dict(datatype=Datatype.MISC))

        else:
            vals_sorted_by_occurence = list(column.value_counts().index)
            stats.update(dict(
                datatype=Datatype.CATEGORICAL,
                unique_vals=vals_sorted_by_occurence,
                num_unique=len(column.unique())
            ))

    else:

        percentiles = list(column.quantile(q=[0.1 * i for i in range(11)]))

        stats.update(dict(
            max=column.max(),
            min=column.min(),
            mean=column.mean(),
            num_unique=len(column.unique()),
            percentiles=percentiles,
        ))

        if column.dtype == int:
            stats.update(dict(datatype=Datatype.INT))

        else:
            stats.update(dict(datatype=Datatype.FLOAT))

    return stats


def generate_stats(data_dir: str, out_path: str, datasets: str, force=True):
    fn = functools.partial(gen_stats_worker, data_dir=data_dir, out_path=out_path, force=force)

    tasks = [(db.db_name, db.data_folder) for db in dataset_list_dict[datasets]]

    with multiprocessing.Pool() as pool:
        pool.map(fn, tasks)


def gen_stats_worker(args, data_dir: str, out_path: str, force=True):
    try:
        dataset, data_folder = args

        # read the schema file
        os.makedirs(out_path, exist_ok=True)
        column_stats_path = os.path.join(out_path, f'{dataset}_column_statistics.json')
        if os.path.exists(column_stats_path) and not force:
            print(f"Column stats already created: {dataset}")
            return

        joint_column_stats, _ = load_df_tables(dataset, data_dir, data_folder)

        # save to json
        with open(column_stats_path, 'w') as outfile:
            # workaround for numpy and other custom datatypes
            json.dump(joint_column_stats, outfile, cls=CustomEncoder)
    except Exception as e:
        print(args, flush=True)
        traceback.print_exc()
        raise e


def load_df_tables(dataset, data_dir: str, data_folder: str, include_stats_gen: bool = True):
    schema = load_schema_json(dataset)

    # read individual table csvs and derive statistics
    joint_column_stats = dict()
    df_tables = dict()
    for t in schema.tables:
        column_stats_table = dict()

        if dataset == 'financial' and t == 'orders':
            table_dir = os.path.join(data_dir, data_folder, f'order.csv')
        else:
            table_dir = os.path.join(data_dir, data_folder, f'{t}.csv')

        assert os.path.exists(table_dir), f"Could not find table csv {table_dir}"
        print(f"Generating statistics for {t}")

        df_table = pd.read_csv(table_dir, **vars(schema.csv_kwargs))

        for column in df_table.columns:
            # for the udfs we have adjusted some column names which otherwise would not be usable with python
            # e.g. Month/Year -> Month_Year, 2B -> SecondB, 3B -> ThirdB
            translated_c = translate_col_names(dataset, t, column)

            if include_stats_gen:
                column_stats_table[translated_c] = column_stats(df_table[column])

        joint_column_stats[t] = column_stats_table
        df_tables[t] = df_table

    return joint_column_stats, df_tables
