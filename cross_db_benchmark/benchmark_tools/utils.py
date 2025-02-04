import json
import os
from types import SimpleNamespace
from typing import Tuple
import orjson

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


def pullup_udf_in_sql(query: str) -> Tuple[str, bool]:
    # separate query parts
    query = query.strip()
    if query.endswith(';'):
        query = query[:-1]
    select_part = query[query.index('SELECT ') + 7:query.index(' FROM ')]
    select_part = rewrite_select_to_nestsql(select_part)

    if ' WHERE ' in query:
        from_part = query[query.index(' FROM ') + 6:query.index(' WHERE ')]
        where_part = query[query.index(' WHERE ') + 7:]
    else:
        from_part = query[query.index(' FROM ') + 6:]
        where_part = None

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

    no_udf_filters = []
    udf_filters = []

    for filter in filters:
        if filter.startswith('func_'):
            udf_filters.append(filter)
        else:
            no_udf_filters.append(filter)

    if len(no_udf_filters) > 0:
        no_udf_where = ' AND '.join(no_udf_filters)
        no_udf_where = f' WHERE {no_udf_where}'
    else:
        no_udf_where = ''

    assert len(udf_filters) == 1, 'Only one udf filter supported'
    rewritten_udf = rewrite_udf_to_nestsql(udf_filters[0])

    query = f'SELECT {select_part} FROM (SELECT * FROM {from_part}{no_udf_where}) as nested_query WHERE {rewritten_udf};'

    return query, True


def rewrite_table_col(table_col: str) -> str:
    if '.' in table_col:
        table, col = table_col.split('.')
        return f'nested_query.{col}'
    else:
        return f'nested_query.{table_col}'


def rewrite_select_to_nestsql(select_str: str) -> str:
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
                found = True
                if ' + ' in col:
                    rewritten_cols.append(
                        f'{aggr.upper()}({rewrite_table_col(col.split(" + ")[0])} + {rewrite_table_col(col.split(" + ")[1])}')
                elif ' - ' in col:
                    rewritten_cols.append(
                        f'{aggr.upper()}({rewrite_table_col(col.split(" - ")[0])} - {rewrite_table_col(col.split(" - ")[1])}')
                elif ' * ' in col:
                    rewritten_cols.append(
                        f'{aggr.upper()}({rewrite_table_col(col.split(" * ")[0])} * {rewrite_table_col(col.split(" * ")[1])}')
                elif ' / ' in col:
                    rewritten_cols.append(
                        f'{aggr.upper()}({rewrite_table_col(col.split(" / ")[0])} / {rewrite_table_col(col.split(" / ")[1])}')
                else:
                    try:
                        suffix = col.split('.')[1]
                    except IndexError as e:
                        print(f'Could not split {col} \n {select_str}')
                        raise e
                    rewritten_cols.append(f'{aggr.upper()}(nested_query.{suffix}')

                break

        if found:
            continue

        tc_split = col.split('.')
        assert len(tc_split) == 2, f'Could not split {col}'
        table, col = tc_split

        rewritten_cols.append(f'nested_query.{col}')

    return ','.join(rewritten_cols)


def rewrite_udf_to_nestsql(udf_str: str) -> str:
    # get table and column
    udf_str = udf_str.strip()
    udf_prefix = udf_str[:udf_str.index('(') + 1]
    udf_suffix = udf_str[udf_str.index(')'):]

    cols = udf_str[udf_str.index('(') + 1:udf_str.index(')')].split(',')
    rewritten_cols = []
    for col in cols:
        col = col.strip()
        if '.' in col:
            table, column = col.split('.')
            rewritten_cols.append(f'nested_query.{column}')
        else:
            rewritten_cols.append(f'nested_query.{col}')

    cols_str = ','.join(rewritten_cols)

    return f'{udf_prefix}{cols_str}{udf_suffix}'
