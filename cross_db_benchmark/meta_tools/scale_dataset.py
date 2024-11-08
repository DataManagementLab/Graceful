import collections
import os
import re
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.utils import load_schema_json, load_column_statistics


def scale_up_dataset(source_dataset, data_dir, target_dir, scale=None, autoscale_tuples=None, autoscale_gb=1,
                     correct_pg_schema=True, scale_individually=False):
    os.makedirs(target_dir, exist_ok=True)
    schema = load_schema_json(source_dataset)

    # adapt foreign keys (extract primary keys and join conditions from schema)
    pg_schema, pg_schema_path, scale_columns = extract_scale_columns(schema, source_dataset)

    column_stats = {k: vars(v) for k, v in vars(load_column_statistics(source_dataset)).items()}
    new_varchar_lengths = dict()

    # find autoscale
    if scale is None:
        if autoscale_tuples is not None:
            total_no_tuples = 0
            for t in schema.tables:
                table_path = os.path.join(data_dir, f'{t}.csv')
                no_tuples = int(os.popen(f'wc -l {table_path}').read().split(' ')[0])
                total_no_tuples += no_tuples

            print("Deriving scale automatically")
            print(f" - current no tuples: {total_no_tuples}")
            print(f" - desired no tuples: {autoscale_tuples}")
            scale = max(autoscale_tuples // total_no_tuples, 1)
            print(f" - scale: {scale}")
            print(total_no_tuples)
        else:
            filesize_gb = get_dataset_size(data_dir, schema)

            print("Deriving scale automatically")
            print(f" - current size: {filesize_gb:.3f}gb")
            print(f" - desired size: {autoscale_gb}gb")
            scale = max(int(autoscale_gb / filesize_gb), 1)
            print(f" - scale: {scale}")

    # custom nan values since NA appears in some datasets and is not nan
    custom_nan_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                         '<NA>', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null']

    # all the table sizes for the standard config w/o scaling to compute the min_max_normalization
    # the largest value in the data will have the value 1, the lowest the value 0
    # the other values have values between 0 and 1 depending on their size
    def min_max_table_norm(schema):
        table_lengths = []
        for t in schema.tables:
            table_path = os.path.join(data_dir, f'{t}.csv')
            table_target_path = os.path.join(target_dir, f'{t}.csv')
            orig_df_table = pd.read_csv(table_path, **vars(schema.csv_kwargs),
                                        keep_default_na=False,
                                        na_values=custom_nan_values)
            table_lengths.append(len(orig_df_table))  # determine the size of the table

        min_max_lengths = []
        for length in table_lengths:
            # apply formula to get the min_max_norm value
            min_max_lengths.append((length - min(table_lengths)) / (max(table_lengths) - min(table_lengths)))

        return min_max_lengths, table_lengths

    # Note: the goal is to scale each table individually; at the end, we want that all the tables have a size
    # which is between 150% and 250% of the largest table in the database
    def get_individual_scaling_factors(scale, min_max_lengths, table_lengths):
        # determine the goal interval for the resulting tables
        # here we also use the original scaling factor to scale the largest table
        t_min, t_max = int(max(table_lengths) * 1.5), int(max(table_lengths) * 2.5)
        scaling_factors = []
        for min_max, length in zip(min_max_lengths, table_lengths):
            # use the min_max_norm value to determine the size of the new table in the goal interval
            # this makes sure that all tables will have a size of at least t_min and at most t_max
            new_tab_size = int(min_max * (t_max - t_min) + t_min)
            # compute the individual scaling factor for each table that is later used for scaling
            scale_factor = int(new_tab_size / length) * scale
            scaling_factors.append(scale_factor)
        return scaling_factors

    if scale_individually:
        min_max_lengths, table_lengths = min_max_table_norm(schema)
        scaling_factors = get_individual_scaling_factors(scale, min_max_lengths, table_lengths)

    for t in schema.tables:
        # use the individual scaling factors if scale_individually is true
        # the order in scaling factors is the same as the order in schema.tables
        if scale_individually:
            scale = scaling_factors.pop(0)

        table_path = os.path.join(data_dir, f'{t}.csv')
        table_target_path = os.path.join(target_dir, f'{t}.csv')
        shutil.copyfile(table_path, table_target_path)

        orig_df_table = pd.read_csv(table_path, **vars(schema.csv_kwargs),
                                    keep_default_na=False,
                                    na_values=custom_nan_values, low_memory=False)
        # circumvents bug: pandas reads integer column into float column and then the import to postgres does not
        # work. The issue is that the underlying integer representation does not support nans. Hence, we resort to
        # a nullable integer type.
        conv_int_cols = set()
        for c in orig_df_table.columns:

            if orig_df_table[c].dtype == np.float64:
                nna_mask = ~np.isnan(orig_df_table[c])

                # all int
                if np.all(np.mod(orig_df_table[c][nna_mask], 1) == 0):
                    orig_df_table[c] = orig_df_table[c].astype(pd.Int64Dtype())
                    conv_int_cols.add(c)

            # for mixed columns which have float elements
            # elif orig_df_table[c].dtype == object:
            #     type_set = {type(e) for e in orig_df_table[c].unique()}
            #     if float in type_set:
            #         max_len = max([len(str(c)) for c in orig_df_table[c].unique() if isinstance(c, float)])
            #         new_varchar_lengths[(t, c)] = max_len
            #         print(f"float is in mixed column {c} (new max_len {max_len})")

        print(f"Scaling table {t}")
        for i in tqdm(range(scale - 1)):

            curr_df_table = orig_df_table.copy(deep=True)
            for c in scale_columns[t]:

                if curr_df_table[c].dtype == np.int64 or c in conv_int_cols:
                    # find maximum value in original column and scale up
                    offset = find_numeric_offset(c, column_stats, schema, t)
                    curr_df_table[c] += (i + 1) * offset

                elif curr_df_table[c].dtype == object:
                    curr_df_table[c] += f'_{i}'

                else:
                    print(curr_df_table[c].dtype)
                    raise NotImplementedError

                if i == scale - 2:
                    max_len = curr_df_table[c].astype(str).str.len().max()
                    new_varchar_lengths[(t, c)] = max_len

            # append scaled up values to the original table
            curr_df_table.to_csv(table_target_path, mode='a', header=False, index=False, **vars(schema.csv_kwargs),
                                 na_rep='NULL')

    # adapt lengths of varchars
    if correct_pg_schema:
        # print(new_varchar_lengths)

        # read mysql schema
        new_schema = []
        for table_def in pg_schema:
            if table_def.startswith('"'):
                table_name = table_def.split("\n")[0].strip('"; ')
                for (t, c), vc_length in new_varchar_lengths.items():
                    if t != table_name:
                        continue
                    for search_string, repl_string in [(f'"{c}" varchar\(\d+\)', f'"{c}" varchar({vc_length:.0f})'),
                                                       (f'"{c}" char\(\d+\)', f'"{c}" char({vc_length:.0f})')]:
                        table_def = re.sub(search_string, repl_string, table_def)

            new_schema.append(table_def)

        pg_schema = "DROP TABLE IF EXISTS ".join(new_schema)
        pg_schema = pg_schema.replace('NOT NULL', '')
        with open(pg_schema_path, 'w') as file:
            file.write(pg_schema)
        print(f"Corrected postgres schema definition {pg_schema_path}")


def scale_down_dataset(source_dataset, data_dir, target_dir, max_size=750000):
    os.makedirs(target_dir, exist_ok=True)
    schema = load_schema_json(source_dataset)

    largest_table_size = 0

    # determine the size of the largest table in the database
    for t in schema.tables:
        table_path = os.path.join(data_dir, f'{t}.csv')
        table_target_path = os.path.join(target_dir, f'{t}.csv')
        shutil.copyfile(table_path, table_target_path)
        orig_df_table = pd.read_csv(table_path, **vars(schema.csv_kwargs))
        if len(orig_df_table.index) > largest_table_size:
            largest_table_size = len(orig_df_table.index)

    # determine the number of rows to be removed
    shrinking_factor = max_size / largest_table_size

    for t in schema.tables:

        table_path = os.path.join(data_dir, f'{t}.csv')
        table_target_path = os.path.join(target_dir, f'{t}.csv')
        shutil.copyfile(table_path, table_target_path)

        # custom nan values since NA appears in some datasets and is not nan
        custom_nan_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                             '<NA>', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null']
        orig_df_table = pd.read_csv(table_path, **vars(schema.csv_kwargs),
                                    keep_default_na=False,
                                    na_values=custom_nan_values, low_memory=False)
        # circumvents bug: pandas reads integer column into float column and then the import to postgres does not
        # work. The issue is that the underlying integer representation does not support nans. Hence, we resort to
        # a nullable integer type.
        conv_int_cols = set()
        for c in orig_df_table.columns:

            if orig_df_table[c].dtype == np.float64:
                nna_mask = ~np.isnan(orig_df_table[c])

                # all int
                if np.all(np.mod(orig_df_table[c][nna_mask], 1) == 0):
                    orig_df_table[c] = orig_df_table[c].astype(pd.Int64Dtype())
                    conv_int_cols.add(c)

        # only downscale tables that are larger than max_size
        if len(orig_df_table.index) > max_size:
            # determine the new size of the table based on the overall shrinking factor
            new_size = int(shrinking_factor * len(orig_df_table.index))
            # determine the indices that are going to be removed
            drop_indices = np.random.choice(orig_df_table.index, len(orig_df_table.index) - new_size, replace=False)
            # drop the indices
            orig_df_table = orig_df_table.drop(drop_indices)

        # write the new table to the target directory
        if schema.name in ["ssb", "tpc_h"]:
            orig_df_table.to_csv(table_target_path, mode='w', header=True, index=False, escapechar="\\",
                                 encoding="utf-8", quotechar="\"", sep="|", na_rep='NULL')
        elif schema.name == "imdb":
            orig_df_table.to_csv(table_target_path, mode='w', header=True, index=False, escapechar="\\",
                                 encoding="utf-8", quotechar="\"", na_rep='')
        elif schema.name == "airline":
            orig_df_table.to_csv(table_target_path, mode='w', header=True, index=False, sep="\t", na_rep='')
        else:
            orig_df_table.to_csv(table_target_path, mode='w', header=True, index=False, **vars(schema.csv_kwargs),
                                 na_rep='NULL')


def get_dataset_size(data_dir, schema):
    filesize_b = 0
    for t in schema.tables:
        table_path = os.path.join(data_dir, f'{t}.csv')
        filesize_b += os.path.getsize(table_path)
    filesize_gb = filesize_b / (1024 ** 3)
    return filesize_gb


def find_numeric_offset(c, column_stats, schema, t):
    offset = None
    # if column is referencing, take this offset
    for t_out, col_out, t_in, col_in in schema.relationships:
        if t_out != t:
            continue

        if not isinstance(col_out, list):
            col_out = [col_out]
            col_in = [col_in]
        if c not in col_out:
            continue

        c_idx = col_out.index(c)

        offset = column_stats[t_in][col_in[c_idx]].max + 1
    # take own offset if it is not referencing
    if offset is None:
        offset = column_stats[t][c].max + 1
    return offset


def extract_type(pg_schema, t, c):
    type = None
    for table_def in pg_schema:
        table_def = table_def.strip('"\'').strip()
        for next_char in [' ', '\'', '"', ';']:
            if table_def.startswith(t + next_char):
                for l in table_def.split('\n'):
                    l = l.replace('"', '').replace("'", "").strip()
                    for next_char in [' ', '\'', '"', ';']:
                        if l.startswith(c + next_char):
                            assert type is None, "Could not find unique column definition"
                            type = re.split(" +", l)[1].strip().strip(',')

    assert type is not None, f"Could not find column definition for {t}.{c}"
    return type


def extract_scale_columns(schema, source_dataset, schema_name='postgres.sql'):
    scale_columns = collections.defaultdict(set)
    for table_l, col_l, table_r, col_r in schema.relationships:
        if not isinstance(col_l, list):
            col_l = [col_l]
            col_r = [col_r]

        for table, columns in [(table_l, col_l), (table_r, col_r)]:
            for c in columns:
                scale_columns[table].add(c)
    # find primary keys to adapt
    pg_schema_path = os.path.join('cross_db_benchmark/datasets/', source_dataset, 'schema_sql', schema_name)
    with open(pg_schema_path, 'r') as file:
        pg_schema = file.read()
    if 'DROP TABLE IF EXISTS' in pg_schema:
        pg_schema = pg_schema.split("DROP TABLE IF EXISTS ")
    elif 'drop table if exists' in pg_schema:
        pg_schema = pg_schema.split("drop table if exists ")
    else:
        raise NotImplementedError
    for table_def in pg_schema:
        if table_def.startswith('"'):
            table_name = table_def.split("\n")[0].strip('"; ')

            # first way of specifying PK
            pk_regex = re.finditer('PRIMARY KEY \((\"\S+\"(,|, )?)+\)', table_def)
            for matched_pk in pk_regex:
                pk_columns = [c.strip('"') for c in matched_pk.groups()[0].split(',')]
                for c in pk_columns:
                    scale_columns[table_name].add(c)
            for l in table_def.split('\n'):
                if 'PRIMARY KEY,' in l:
                    pk_alt = l.strip().split(' ')[0].strip('"')
                    scale_columns[table_name].add(pk_alt)

    return pg_schema, pg_schema_path, scale_columns
