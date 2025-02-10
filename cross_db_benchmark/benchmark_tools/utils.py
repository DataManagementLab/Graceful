import json
import os
import random
from collections import defaultdict
from types import SimpleNamespace
from typing import Tuple, Dict
import orjson
import re
import math

def load_schema_json(dataset):
    schema_path = os.path.join('cross_db_benchmark/datasets/', dataset, 'schema.json')
    assert os.path.exists(schema_path), f"Could not find schema.json ({schema_path})"
    schema = load_json(schema_path)

    # apply schema modifications
    if dataset == 'financial':
        assert schema.tables[2] == 'order'
        schema.tables[2] = 'orders'

        assert schema.relationships[5][0] == 'order'
        schema.relationships[5][0] = 'orders'

    return schema


def translate_col_names(dataset: str, table: str, col: str):
    if dataset == 'baseball':
        if col == '2B':
            return 'SecondB'
        elif col == '3B':
            return 'ThirdB'
        else:
            return col
    elif dataset == 'seznam':
        if col == 'Month/Year':
            return 'Month_Year'
        else:
            return col
    else:
        return col


def rewrite_query_to_schema(sql_query: str, dataset: str):
    if dataset == 'baseball':
        sql_query = sql_query.replace('\"2B\"', '\"SecondB\"')
        sql_query = sql_query.replace('\"3B\"', '\"ThirdB\"')
    elif dataset == 'seznam':
        sql_query = sql_query.replace('\"Month/Year\"', '\"Month_Year\"')

    return sql_query


def orig_col_names(dataset: str, table: str, col: str):
    if dataset == 'baseball':
        if col == 'SecondB':
            return '2B'
        elif col == 'ThirdB':
            return '3B'
        else:
            return col
    elif dataset == 'seznam':
        if col == 'Month_Year':
            return 'Month/Year'
        else:
            return col
    else:
        return col


def load_column_statistics(col_stats_dir: str, dataset: str, namespace=True):
    path = os.path.join(col_stats_dir, f'{dataset}_column_statistics.json')
    assert os.path.exists(path), f"Could not find file ({path})"
    return load_json(path, namespace=namespace)


def load_string_statistics(col_stats_dir: str, dataset: str, namespace=True):
    path = os.path.join(col_stats_dir, f'{dataset}_string_statistics.json')
    assert os.path.exists(path), f"Could not find file ({path})"
    return load_json(path, namespace=namespace)


def load_json(path, namespace=True):
    with open(path) as json_file:
        if namespace:
            json_obj = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        else:
            json_obj = orjson.loads(json_file.read())
            # json_obj = json.load(json_file)
    return json_obj


def load_schema_sql(dataset, sql_filename):
    raise Exception('please use zs_dbms_setup repo for this')
    sql_path = os.path.join('cross_db_benchmark/datasets/', dataset, 'schema_sql', sql_filename)
    assert os.path.exists(sql_path), f"Could not find schema.sql ({sql_path})"
    with open(sql_path, 'r') as file:
        data = file.read().replace('\n', '')
    return data

def is_join_cond(filter: str) -> bool:
    join_cond_pattern = re.compile(r'\b"?\w+"?\."?\w+"?\s*=\s*"?\w+"?\."?\w+"?\b')

    return bool(join_cond_pattern.search(filter))

def parse_join_cond(filter: str) -> Tuple[str, str]:
    assert is_join_cond(filter), f'Filter is not a join condition: {filter}'

    match = re.match(r'"?(\w+)"?\."?(\w+)"?\s*=\s*"?(\w+)"?\."?(\w+)"?', filter.strip())
    assert match is not None, f'Could not parse join condition: {filter}'
    table1, column1, table2, column2 = match.groups()
    return table1, column1, table2, column2

def extract_table_from_udf_filter(udf_filter: str) -> str:
    udf_filter = udf_filter.strip()
    assert udf_filter.startswith('func_'), 'Filter is not a udf filter'

    # filter has form of func_XXX(table1.col1, ..., tableX.colX)>13

    # extract everything between brackets
    udf_tables = udf_filter[udf_filter.index('(')+1:udf_filter.index(')')]
    udf_tables = udf_tables.split(',')
    udf_tables = [table.split('.')[0] for table in udf_tables]

    # remove leading & trailing "
    udf_tables = [table.strip().strip('"') for table in udf_tables]

    assert len(set(udf_tables)) == 1, 'Only one table supported'
    return udf_tables[0]

def extract_table_from_filter(filter: str) -> str:
    t =  filter.split('.')[0]
    # remove "
    t = t.strip().strip('"')
    return t

def extract_table_from_from_clause(from_clause):
    join_conds = []

    if ' JOIN ' in from_clause and ' ON ' in from_clause:
        # parse join on
        splits = from_clause.split(' JOIN ')
        tables = [splits[0].strip()]
        for s in splits[1:]:
            table, join_cond = s.split(' ON ')
            table = table.strip()
            tables.append(table)

            join_cond = parse_join_cond(join_cond)
            join_conds.append(join_cond)
    else:
        # split by ,
        tables = from_clause.split(',')

    tables = [t.strip().strip('"') for t in tables]

    return tables, join_conds

def pullup_udf_in_sql(query: str, intermed_pos:bool=False, random_seed:int=None) -> Tuple[str, bool]:
    if random_seed is not None:
        random_generator = random.Random(random_seed)
    else:
        random_generator = random

    # separate query parts
    query = query.strip()
    if query.endswith(';'):
        query = query[:-1]

    # dict mapping old col names to new col names
    col_renamings_dict = dict()

    select_part = query[query.index('SELECT ') + 7:query.index(' FROM ')]

    # check func_123 in select part
    if re.search(r'\bfunc_\d+\(', select_part):
        return query, False

    select_part = rewrite_select_to_nestsql(select_part, prefix_cols_with_table = intermed_pos, rewriting_cols_dict=col_renamings_dict)

    if ' WHERE ' in query:
        from_part = query[query.index(' FROM ') + 6:query.index(' WHERE ')]
        where_part = query[query.index(' WHERE ') + 7:]
    else:
        from_part = query[query.index(' FROM ') + 6:]
        where_part = None

    # split from part into tables
    from_tables, join_conds = extract_table_from_from_clause(from_part)

    # check whether there is a filter in query
    if where_part is None:
        return query, False

    # check whether udf is involved in filter
    where_part = where_part.strip()
    if not where_part.startswith('func_') and ' func_' not in where_part:
        return query, False

    assert ' OR ' not in where_part, 'OR not supported yet'

    # split filters
    filters = where_part.split(' AND ')

    no_udf_filters_dict = defaultdict(list) # dict mapping table name to involved filters
    udf_filters_dict = defaultdict(list) # dict mapping table name to involved filters

    for filter in filters:
        if filter.startswith('func_'):
            udf_filters_dict[extract_table_from_udf_filter(filter)].append(filter)
        elif is_join_cond(filter):
            jc = parse_join_cond(filter)
            join_conds.append(jc)
        else:
            no_udf_filters_dict[extract_table_from_filter(filter)].append(filter)

    # create join lookup
    join_lookup = defaultdict(list)
    for jc in join_conds:
        join_lookup[jc[0]].append(jc)
        join_lookup[jc[2]].append(jc)

    all_tables = set(no_udf_filters_dict.keys()).union(set(udf_filters_dict.keys())).union(set(join_lookup.keys()))

    assert len(udf_filters_dict) == 1, 'Only one udf filter supported'

    no_udf_filters = [f for table, filters in no_udf_filters_dict.items() for f in filters]
    udf_filters = [f for table, filters in udf_filters_dict.items() for f in filters]

    # rewrite udf
    assert len(udf_filters) == 1, 'Only one udf filter supported'
    rewritten_udf = rewrite_udf_to_nestsql(udf_filters[0], prefix_cols_with_table=intermed_pos, col_renamings_dict=col_renamings_dict)

    # split in table below UDF and above UDF
    if intermed_pos:
        # make sure that all tables from from clause are in join conditions
        assert set(from_tables) == all_tables, f'Not all tables in from clause are in join conditions - not supported for intermed positioning: {query} \n from tables: {from_tables} \n all tables: {all_tables}'

        # num tables below - num_joins/2+1
        num_tables_below = math.ceil((len(all_tables)-1) / 2) +1

        below_tables = [list(udf_filters_dict.keys())[0]]

        # perform radom walks starting from UDF table to find tables below
        while len(below_tables) < num_tables_below:
            # randomly shuffle below_tables
            random_generator.shuffle(below_tables)

            join_candidates = []
            for table in below_tables:
                # get tables who can be joined with table and are not in below_tables
                for jc in join_lookup[table]:
                    if jc[0] == table:
                        if jc[2] not in below_tables:
                            join_candidates.append(jc[2])
                    elif jc[2] == table:
                        if jc[0] not in below_tables:
                            join_candidates.append(jc[0])

            assert len(join_candidates) > 0, 'Could not find join candidates'
            below_tables.append(random_generator.choice(join_candidates))

        above_tables = list(all_tables - set(below_tables))
        above_tables = sorted(above_tables)

        # assemble from below
        from_below = ','.join(below_tables)

        # assemble join tables above UDF
        from_above = ','.join(above_tables)
        # add leading ,
        if from_above != '':
            from_above = f',{from_above}'

        # assemble join conditions above UDF
        join_conds_above = [j for j in join_conds if j[0] in above_tables or j[2] in above_tables]
        join_conds_below = [j for j in join_conds if j[0] in below_tables and j[2] in below_tables]

        join_conds_below = [f'{j[0]}.{j[1]}={j[2]}.{j[3]}' for j in join_conds_below]

        join_conds_above_str = []
        for j in join_conds_above:
            if j[0] in below_tables:
                new_col_name = f'{j[0]}_{j[1]}'
                col_renamings_dict[(j[0], j[1])] = new_col_name
                join_conds_above_str.append(f'nested_query.{new_col_name }={j[2]}.{j[3]}')
            elif j[2] in below_tables:
                new_col_name = f'{j[2]}_{j[3]}'
                col_renamings_dict[(j[2], j[3])] = new_col_name
                join_conds_above_str.append(f'{j[0]}.{j[1]}=nested_query.{new_col_name}')
            else:
                join_conds_above_str.append(f'{j[0]}.{j[1]}={j[2]}.{j[3]}')

        # assemble filter below
        filters_below = [f for table, filters in no_udf_filters_dict.items() if table in below_tables for f in filters]
        filters_below = join_conds_below + filters_below # add join_conds to list
        if len(filters_below) > 0:
            where_below = ' AND '.join(filters_below)
            where_below = f' WHERE {where_below}'
        else:
            where_below = ''

        # assemble filters above (on other tables not in subquery)
        filters_above = [f for table, filters in no_udf_filters_dict.items() if table in above_tables for f in filters]
        filters_above = join_conds_above_str + filters_above # add join_conds to list

        # assemble join conditions above
        if len(filters_above) > 0:
            conds_above = ' AND '.join(filters_above)
            conds_above = f' AND {conds_above}'
        else:
            conds_above = ''

        # assemble renaming string
        renaming_strings = []
        for old_col, new_col in col_renamings_dict.items():
            renaming_strings.append(f'{old_col[0]}.{old_col[1]} AS {new_col}')
        renaming_string = ','.join(renaming_strings)

        if renaming_string == '':
            renaming_string = '*'

        subquery = f'SELECT {renaming_string} FROM {from_below}{where_below}'
        query = f'SELECT {select_part} FROM ({subquery}) as nested_query{from_above} WHERE {rewritten_udf}{conds_above};'

    else:
        no_udf_filters = [f'{j[0]}.{j[1]}={j[2]}.{j[3]}' for j in join_conds] + no_udf_filters
        if len(no_udf_filters) > 0:
            no_udf_where = ' AND '.join(no_udf_filters)
            no_udf_where = f' WHERE {no_udf_where}'
        else:
            no_udf_where = ''

        query = f'SELECT {select_part} FROM (SELECT * FROM {from_part}{no_udf_where}) as nested_query WHERE {rewritten_udf};'

    return query, True


def rewrite_table_col(table_col: str, prefix_cols_with_table:bool, rewriting_cols_dict:Dict) -> str:
    if table_col.startswith('func_'):
        return rewrite_udf_to_nestsql(table_col, prefix_cols_with_table, rewriting_cols_dict)

    if '.' in table_col:
        table, col = table_col.split('.')
        table = table.strip().strip('"')
        col = col.strip().strip('"')
        if prefix_cols_with_table:
            assert '"' not in table, table_col
            rewriting_cols_dict[(table, col)] = f'{table}_{col}'
            return f'nested_query.{table}_{col}'
        else:
            return f'nested_query.{col}'
    else:
        return f'nested_query.{table_col}'


def rewrite_select_to_nestsql(select_str: str, prefix_cols_with_table:bool, rewriting_cols_dict:Dict) -> str:
    cols = select_str.split(',')

    rewritten_cols = []
    for col in cols:
        col = col.strip()

        if col.lower().startswith('count(*)') or col == '*':
            rewritten_cols.append(col)
            continue

        found = False
        for aggr in ['SUM', 'COUNT', 'AVG', 'sum', 'count', 'avg']:
            if col.startswith(aggr):
                # extract closing part e.g. ) as count
                suffix = ''
                if ' as ' in col:
                    col, suffix = col.split(' as ')
                    suffix = f' as {suffix}'

                assert col.endswith(')'), f'Could not split {col}'
                col = col[len(aggr) + 1:len(col)-1]  # remove aggr and ( and closing )
                found = True
                if ' + ' in col:
                    rewritten_cols.append(
                        f'{aggr.upper()}({rewrite_table_col(col.split(" + ")[0], prefix_cols_with_table, rewriting_cols_dict)} + {rewrite_table_col(col.split(" + ")[1],prefix_cols_with_table, rewriting_cols_dict)})')
                elif ' - ' in col:
                    rewritten_cols.append(
                        f'{aggr.upper()}({rewrite_table_col(col.split(" - ")[0],prefix_cols_with_table, rewriting_cols_dict)} - {rewrite_table_col(col.split(" - ")[1],prefix_cols_with_table, rewriting_cols_dict)})')
                elif ' * ' in col:
                    rewritten_cols.append(
                        f'{aggr.upper()}({rewrite_table_col(col.split(" * ")[0],prefix_cols_with_table, rewriting_cols_dict)} * {rewrite_table_col(col.split(" * ")[1],prefix_cols_with_table, rewriting_cols_dict)})')
                elif ' / ' in col:
                    rewritten_cols.append(
                        f'{aggr.upper()}({rewrite_table_col(col.split(" / ")[0],prefix_cols_with_table, rewriting_cols_dict)} / {rewrite_table_col(col.split(" / ")[1],prefix_cols_with_table, rewriting_cols_dict)})')
                else:
                    assert '.' in col, f'Could not split {col}'

                    c = rewrite_table_col(col,prefix_cols_with_table, rewriting_cols_dict)

                    rewritten_cols.append(f'{aggr.upper()}({c}){suffix}')

                break

        if found:
            continue

        rewritten_cols.append(rewrite_table_col(col,prefix_cols_with_table, rewriting_cols_dict))

    return ','.join(rewritten_cols)


def rewrite_udf_to_nestsql(udf_str: str, prefix_cols_with_table:bool, col_renamings_dict:Dict) -> str:
    # get table and column
    udf_str = udf_str.strip()
    udf_prefix = udf_str[:udf_str.index('(') + 1]
    udf_suffix = udf_str[udf_str.index(')'):]

    cols = udf_str[udf_str.index('(') + 1:udf_str.index(')')].split(',')
    rewritten_cols = []

    for col in cols:
        col = col.strip()
        rewritten_cols.append(rewrite_table_col(col,prefix_cols_with_table, col_renamings_dict))

    cols_str = ','.join(rewritten_cols)

    return f'{udf_prefix}{cols_str}{udf_suffix}'
