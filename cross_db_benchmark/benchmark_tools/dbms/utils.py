import math
import re


def plan_statistics(plan_op, tables=None, filter_columns=None, operators=None, skip_columns=False, conv_to_dict=False):
    if tables is None:
        tables = set()
    if operators is None:
        operators = set()
    if filter_columns is None:
        filter_columns = set()

    if not isinstance(plan_op, dict):
        params = plan_op.plan_parameters
        children = plan_op.children
    else:
        params = plan_op['plan_parameters']
        children = plan_op['children']

    if conv_to_dict:
        params = vars(params)

    if 'table' in params:
        tables.add(params['table'])
    if 'op_name' in params:
        operators.add(params['op_name'])
    if 'filter_columns' in params and not skip_columns:
        list_columns(params['filter_columns'], filter_columns)

    for c in children:
        plan_statistics(c, tables=tables, filter_columns=filter_columns, operators=operators, skip_columns=skip_columns,
                        conv_to_dict=conv_to_dict)

    return tables, filter_columns, operators


def child_prod(p, feature_name, default=1):
    child_feat = [c.plan_parameters.get(feature_name) for c in p.children
                  if c.plan_parameters.get(feature_name) is not None]
    if len(child_feat) == 0:
        return default
    return math.prod(child_feat)


def list_columns(n, columns):
    if isinstance(n, dict):
        columns.add((n['column'], n['operator']))
        children = n['children']
    else:
        columns.add((n.column, n.operator))
        children = n.children

    for c in children:
        list_columns(c, columns)


def remove_cast_nesting(col: str, offset: int = 0):
    txt = ''
    opened_brackets = 0

    if 'CAST(' not in col:
        return col, len(col)

    while offset < len(col):
        if col[offset:].startswith('CAST('):
            res, offset = remove_cast_nesting(col, offset + 5)
            txt += res
        elif col[offset] == '(':
            opened_brackets += 1
            txt += col[offset]
        elif col[offset] == ')' and opened_brackets == 0:
            # this is a bracket closing the cast
            if ' AS ' in txt:
                txt = txt.split(' AS ')[0]
            return txt, offset
        elif col[offset] == ')':
            # this is a bracket closing a nested expression
            opened_brackets -= 1
            txt += col[offset]
        else:
            txt += col[offset]

        offset += 1

    return txt, offset


def extract_udf_params_from_str(udf_str: str):
    # remove cast nesting
    udf_str, _ = remove_cast_nesting(udf_str)

    # filter out udf params
    udf_params_regex = r'func_[0-9]+\(([a-zA-Z0-9\,\_\s]+)\)'
    match = re.search(udf_params_regex, udf_str)
    assert match is not None, f'Could not extract udf params from {udf_str}'
    assert len(match.groups()) == 1, f'Could not extract udf params from {udf_str}'
    udf_str = match.groups()[0]

    # split by comma and remove whitespaces
    params = udf_str.split(',')
    params = [p.strip() for p in params]
    return params
