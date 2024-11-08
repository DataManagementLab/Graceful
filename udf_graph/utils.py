from typing import List

from deepdb.inference import FilterCond

from cross_db_benchmark.benchmark_tools.postgres.parse_filter import parse_filter


def q_err(cardinality_predict, cardinality_true):
    if cardinality_predict == 0 and cardinality_true == 0:
        q_error = 1.
    elif cardinality_true == 0:
        q_error = 1.
    elif cardinality_predict == 0:
        q_error = cardinality_true
    else:
        q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
    return q_error


def extract_where_conditions(raw_query: str, target_table: str) -> List[FilterCond]:
    """
    Extracts the where conditions on the target table from the raw query. Only works for queries where join conditions are stated as JOIN ON and not in the WHERE clause
    :param raw_query: sql query
    :param target_table: name of the table to extract the conditions for
    :return: conditions
    """

    # check whether we have a nested query - i.e. only focus on inner query
    if ' as nested_query' in raw_query:
        raw_query = raw_query.split(' as nested_query')[0]
        raw_query = raw_query.split('FROM (SELECT ')[1]
        raw_query = 'SELECT ' + raw_query

    # retrieve where clause
    if ' WHERE ' not in raw_query:
        return []

    from_clause = raw_query.split(' FROM ')[1]
    from_clause = from_clause.split(' WHERE ')[0]

    # extract tables from from clause
    tables = []
    for table in from_clause.split(','):
        if ' AS ' in table:
            table = table.split(' AS ')[0]
        tables.append(table.strip().strip('"'))

    where_splits = raw_query.split(' WHERE ')
    assert len(where_splits) == 2, f"Multiple WHERE clauses found in raw query: {raw_query}"
    where_clause = where_splits[1]

    # strip group by clause if it exists
    if 'GROUP BY' in raw_query:
        where_clause = where_clause.split(' GROUP BY ')[0]

    # remove trailing semicolon
    if where_clause.endswith(';'):
        where_clause = where_clause[:-1]

    # extract conditions on the target table
    conditions = []
    assert ' OR ' not in where_clause, f"OR condition found in where clause: {where_clause}"
    for cond in where_clause.split(' AND '):
        # remove leading and trailing white spaces
        cond = cond.strip()

        # skip function calls
        if cond.startswith('func_'):
            continue

        # add conditions on the target table
        if target_table is None:
            if len(tables) > 1:
                assert '.' in cond, f"No table name found in condition: {cond}"
            conditions.append(cond)
        elif f'{target_table}.' in cond:
            assert '.' in cond, f"No table name found in condition: {cond}"
            conditions.append(cond)
        elif f'"{target_table}".' in cond:
            assert '.' in cond, f"No table name found in condition: {cond}"
            conditions.append(cond)

    # parse conditions to FilterCond object
    filter_cond_list = []
    for cond in conditions:
        tmp = parse_filter(f'({cond})', duckdb=True, sql=raw_query)
        assert len(tmp.children) == 0, f"Condition has children: {tmp}"

        if len(tables) > 1:
            assert len(tmp.column) == 2, f"expected table,column tuple: {tmp}"

        if len(tmp.column) == 2:
            filter_cond_list.append(
                FilterCond(table_name=tmp.column[0].strip('"'), column_name=tmp.column[1].strip('"'),
                           operator=str(tmp.operator), value=tmp.literal))
        else:
            filter_cond_list.append(
                FilterCond(table_name=tables[0], column_name=tmp.column[0].strip('"'),
                           operator=str(tmp.operator), value=tmp.literal))

    return filter_cond_list
