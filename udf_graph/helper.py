from typing import Tuple, List, Any, Union, Dict, NamedTuple

import gast
from deepdb.inference import FilterCond

from cross_db_benchmark.benchmark_tools.generate_workload import LogicalOperator
from cross_db_benchmark.benchmark_tools.postgres.parse_filter import PredicateNode


def extract_from_BinOp(ast_subtree, alias2lib_map, func2lib_map, libnames):
    # variable store
    to_visit = []
    ops = []
    datatypes = []
    lib_funcs = []
    used_vars_lib_calls = []
    used_vars_arith_exp = []

    if type(ast_subtree) == gast.gast.BinOp:  # add the starting node to the list of nodes we want to visit
        to_visit.append(ast_subtree)

    while len(to_visit) > 0:
        curr_elem = to_visit.pop()
        left_child = curr_elem.left
        right_child = curr_elem.right
        ops.append(type(curr_elem.op))  # Collect information on the operation associated with the BinOp

        # investigate the left child of BinOp
        # If it is also a BinOp, then add it the list of nodes we need to visit
        # If not, then collect the data for featurization
        if type(left_child) != gast.gast.BinOp:
            if type(left_child) == gast.gast.Constant:
                datatypes.append(type(left_child.value))
            elif type(left_child) == gast.gast.Call:
                lib, args = get_lib_func(left_child, alias2lib_map, func2lib_map, libnames)
                lib_funcs += lib
                used_vars_lib_calls += args
            elif type(left_child) == gast.gast.Name:
                used_vars_arith_exp.append(left_child.id)

        else:
            to_visit.append(left_child)
        # investigate the right child of BinOp
        # If it is also a BinOp, then add it the list of nodes we need to visit
        # If not, then collect the data for featurization
        if type(right_child) != gast.gast.BinOp:
            if type(right_child) == gast.gast.Constant:
                datatypes.append(type(right_child.value))
            elif type(right_child) == gast.gast.Call:
                lib, args = get_lib_func(right_child, alias2lib_map, func2lib_map, libnames)
                lib_funcs += lib
                used_vars_lib_calls += args
            elif type(right_child) == gast.gast.Name:
                used_vars_arith_exp.append(right_child.id)
        else:
            to_visit.append(right_child)

    assert len(lib_funcs) == len(used_vars_lib_calls)
    return ops, datatypes, lib_funcs, used_vars_lib_calls, used_vars_arith_exp


def extract_from_AssignExpr(ast_subtree: Union[gast.gast.Expr, gast.gast.Assign], alias2lib_map: Dict[str, str],
                            func2lib_map: Dict[str, str], libnames: List[str]):
    libs = []
    args = []
    if hasattr(ast_subtree, 'value') and type(ast_subtree.value) != gast.gast.Constant and type(
            ast_subtree.value) != gast.gast.Name:
        lib, arg = get_lib_func(ast_subtree.value, alias2lib_map, func2lib_map, libnames)
        libs += lib
        args += arg
    return [], [], libs, args


def get_lib_func(ast_subtree, alias2lib_map: Dict[str, str], func2lib_map: Dict[str, str], libnames: List[str]) -> \
        Tuple[List[str], List[List]]:
    to_visit = [ast_subtree]
    out_lst = []
    arg_lst = []

    assert type(
        ast_subtree) == gast.gast.Call, f"ast_subtree is not of type Call: {type(ast_subtree)} / {ast_subtree.__dict__}"

    while len(to_visit) > 0:
        curr_call = to_visit.pop(0)
        curr_args = []
        try:
            current = curr_call.func
        except AttributeError as e:
            print(ast_subtree)
            raise e

        for arg in curr_call.args:  # for handling nested function calls => e.g., math.sqrt(math.log(...))
            if type(arg) == gast.gast.Call:
                to_visit.append(arg)
            if type(arg) == gast.gast.Name:
                curr_args.append(arg.id)

        if type(current.value) == gast.gast.Name:
            if current.value.id not in libnames:
                if not hasattr(current.value, "value"):
                    # variable name
                    curr_args.append(current.value.id)
                    out_lst.append(current.attr)
            else:
                # here we have a call like math.sqrt
                out_lst.append(str(current.value.id) + "." + current.attr)
        elif type(current.value) == gast.gast.Attribute:
            func_name = "." + current.attr
            lib_path = []
            while hasattr(current, 'value'):
                current = current.value
                if hasattr(current, "attr"):
                    lib_path.append(current.attr)
            lib_path = reversed(lib_path)
            lib_path_str = ""
            for elem in lib_path:
                lib_path_str += "." + elem
            lib_name = current.id
            if current.id in alias2lib_map.keys():
                lib_name = alias2lib_map[current.id]
            out_lst.append(lib_name + lib_path_str + func_name)
        elif type(current.value) == gast.gast.Constant:
            out_lst.append(current.attr)
        elif type(current.value) == gast.gast.BinOp:
            # This parsing is very tied to the current nature of how string vars are used in our fuzzer
            # We may need to change this when we change the fuzzer or when we use real world data
            out_lst.append(current.attr)
            curr_args.append(current.value.left.right.id)
        # if no variables/arguments could be found, then just insert None to keep the matching between calls and variables stable
        if len(curr_args) == 0:
            curr_args.append(None)

        arg_lst.append(curr_args)

    assert len(out_lst) == len(arg_lst)
    return list(reversed(out_lst)), list(reversed(
        arg_lst))  # reverse the list to have the right structure for creating the nodes => innermost call at first


def extract_from_BoolOp(ast_subtree: gast.gast.BoolOp) -> [List[Any], List[gast.gast.cmpop],
                                                           List[Any],
                                                           List[str]]:  # ast_subtree is of type gast.gast.BoolOp
    bool_connectors = []
    cmops = []
    dtypes = []
    varnames = []
    to_visit = []

    to_visit.append(ast_subtree)

    while len(to_visit) > 0:
        currBoolOp = to_visit.pop(0)
        bool_connectors.append(type(currBoolOp.op))
        for val in currBoolOp.values:
            if type(val) == gast.gast.BoolOp:
                to_visit.append(val)
            elif type(val) == gast.gast.Compare:  # call subroutine for extracting data from a compare statement
                cmop, dtype, cmp_varnames = extract_from_Compare(val)
                cmops.append(cmop)
                dtypes.append(dtype)
                varnames.extend(cmp_varnames)

    return bool_connectors, cmops, dtypes, varnames


def extract_from_Compare(ast_subtree: gast.gast.Compare) -> Tuple[
    gast.gast.cmpop, Any, List[str]]:  # ast_subtree is of type gast.gast.Compare
    # determine the compare operation
    assert len(ast_subtree.ops) == 1, f"Compare statement has more than one compare operation: {ast_subtree.ops}"
    assert len(
        ast_subtree.comparators) == 1, f"Compare statement has more than one comparator: {ast_subtree.comparators}"
    cmop = type(ast_subtree.ops[0])
    varnames = []
    dtype = None

    # determine datatype of the comparison and the varname that is used within the comparison
    if (type(ast_subtree.left) == gast.gast.Constant):
        # variable part of comparison is stored in comparators
        # hence we have the following structure, e.g.: 10 > a
        dtype = type(ast_subtree.left.value)
        varnames.append(ast_subtree.comparators[0].id)

    elif (type(ast_subtree.left) == gast.gast.Name):
        if (type(ast_subtree.comparators[0]) != gast.gast.UnaryOp):
            if hasattr(ast_subtree.comparators[0], "value"):
                # variable part of comparison is stored in comparators
                # hence we have the following structure, e.g.: a > 10
                dtype = type(ast_subtree.comparators[0].value)
                varnames.append(ast_subtree.left.id)
            else:
                varnames.append(ast_subtree.comparators[0].id)
    return cmop, dtype, varnames


class JoinCond(NamedTuple):
    table_name1: str
    column_name1: str
    operator: str
    table_name2: str
    column_name2: str


def join_cond_to_sql(join_cond: JoinCond) -> str:
    return f"{join_cond.table_name1}.{join_cond.column_name1} {join_cond.operator} {join_cond.table_name2}.{join_cond.column_name2}"


def reconstruct_table_name(column_name: str, table_names: List[str], database_statistics: Dict,
                           allow_none_return: bool = False) -> str:
    # reconstruct table name
    table_matches = []
    if isinstance(database_statistics, dict):
        for entry in database_statistics['column_stats']:
            if column_name == entry['column_name'] and entry['table_name'] in table_names:
                table_matches.append(entry['table_name'])
    else:
        for entry in database_statistics.column_stats:
            if column_name == entry.column_name and entry.table_name in table_names:
                table_matches.append(entry.table_name)

    if len(table_matches) == 0 and allow_none_return:
        return None

    assert len(
        table_matches) >= 1, f"Could not find table for column {column_name}: {table_matches} \n{table_names}\n{database_statistics}"

    # it can also happen that the column name is not unique and multiple table candidates are found.
    return table_matches[0]


def parse_join_condition(join_condition: str, left_tables: List[str], right_tables: List[str],
                         database_statistics: Dict,
                         schema_relationships: List[Tuple[str, List, str, List]] = None) -> JoinCond:
    # check whether join condition is prepended with INNER / ...
    assert join_condition.startswith('INNER\n'), f"Join condition does not start with INNER: {join_condition}"
    join_condition = join_condition.replace('INNER\n', '').strip()

    # split join condition
    assert ' = ' in join_condition, f"Join condition does not contain =: {join_condition}"
    split = join_condition.split(' = ')
    assert len(split) == 2, f"Join condition could not be split: {join_condition}"
    left = split[0]
    right = split[1]

    # check if left and right are in the format table.column
    left_split = left.split('.')
    right_split = right.split('.')

    if len(left_split) == 2:
        table1 = left_split[0]
        column1 = left_split[1]
    else:
        # reconstruct table name
        table1 = reconstruct_table_name(left, left_tables, database_statistics,
                                        allow_none_return=schema_relationships is not None)

        column1 = left

    if len(right_split) == 2:
        table2 = right_split[0]
        column2 = right_split[1]
    else:
        # reconstruct table name
        table2 = reconstruct_table_name(right, right_tables, database_statistics,
                                        allow_none_return=schema_relationships is not None)
        column2 = right

    if table1 is None or table2 is None:
        # try to reconstruct from schema relationships
        assert len(left_tables) == 1 and len(
            right_tables) == 1, f"Multiple tables found: {left_tables} / {right_tables}"

        found = False
        for t1, c1s, t2, c2s in schema_relationships:
            if t1 in left_tables and t2 in right_tables:
                found = True

                table1 = t1
                column1 = c1s[0]
                table2 = t2
                column2 = c2s[0]
            elif t2 in left_tables and t1 in right_tables:
                found = True
                table1 = t2
                column1 = c2s[0]
                table2 = t1
                column2 = c1s[0]
            else:
                continue

        assert found, f"Could not reconstruct table names: {table1} / {table2} / {left} / {right} / {left_tables} / {right_tables} / {schema_relationships}"

    assert isinstance(table1, str) and isinstance(table2, str), f"Table names are not strings: {table1} / {table2}"
    return JoinCond(table_name1=table1, column_name1=column1, operator='=', table_name2=table2, column_name2=column2)


def extract_filter_conditions_recursively(filter_col: Union[PredicateNode, Dict], database_statistics) -> List[
    FilterCond]:
    if isinstance(filter_col, PredicateNode):
        dict_filter_col = filter_col.to_dict()
    else:
        dict_filter_col = filter_col

    if len(dict_filter_col['children']) > 0 or dict_filter_col['operator'] in [LogicalOperator.AND, 'AND']:
        # node has children i.e. a logical operator
        # + weird case where a AND node has no children
        assert dict_filter_col['operator'] in ['AND',
                                               LogicalOperator.AND], f"Only AND operator is supported. Found {dict_filter_col['operator']}."
        filter_conditions = []

        for child in dict_filter_col['children']:
            filter_conditions.extend(
                extract_filter_conditions_recursively(child, database_statistics=database_statistics))

        return filter_conditions
    elif 'udf_name' in dict_filter_col and dict_filter_col['udf_name'] is not None:
        # skip udf filters - we cannot calculate the selectivity for them
        return []
    else:
        # node has no children i.e. a filter condition
        if isinstance(database_statistics, dict):
            try:
                col_stats = database_statistics['column_stats'][dict_filter_col['column']]
            except Exception as e:
                print(dict_filter_col)
                raise e
            table_name = col_stats['table_name']
            column_name = col_stats['column_name']
        else:
            col_stats = database_statistics.column_stats[dict_filter_col['column']]
            table_name = col_stats.table_name
            column_name = col_stats.column_name

        assert column_name not in ['2B', '3B', 'class_',
                                   'Month/Year'], f"Column name is a reserved keyword: {column_name}\n{filter_col}"

        if dict_filter_col['literal'] == '' and dict_filter_col['operator'] == '=':
            dict_filter_col['literal'] = '\' \''

        return [FilterCond(table_name=table_name, column_name=column_name, operator=str(dict_filter_col['operator']),
                           value=dict_filter_col['literal'])]


def extract_filter_join_conds_below_udf_from_plan(query_plan_node, database_statistics: Dict, schema_relationships,
                                                  udf_found: bool = False,
                                                  assume_lazy_dbms: bool = False) -> Tuple[
    List[FilterCond], List[JoinCond], List[str]]:
    # udf found
    udf_found_in_this_node = False
    if 'udf_table' in query_plan_node['plan_parameters']:
        udf_found_in_this_node = True
        udf_found = True

    filter_conds = []
    join_conds = []
    tables = []

    # recursively search for filter conditions in the children of this node
    for child in query_plan_node['children']:
        child_filter_conds, child_join_conds, child_tables = extract_filter_join_conds_below_udf_from_plan(child,
                                                                                                           database_statistics=database_statistics,
                                                                                                           schema_relationships=schema_relationships,
                                                                                                           udf_found=udf_found)

        filter_conds.extend(child_filter_conds)
        join_conds.extend(child_join_conds)
        tables.append(child_tables)

    # only extract filter conditions (in case assume_lazy_dbms is active) if udf was not found in this node
    if assume_lazy_dbms or not udf_found_in_this_node:
        if udf_found:
            if 'filter_columns' in query_plan_node['plan_parameters']:
                # search for filter conditions in this node
                try:
                    filters = extract_filter_conditions_recursively(
                        query_plan_node['plan_parameters']['filter_columns'],
                        database_statistics)
                except Exception as e:
                    print(query_plan_node)
                    raise e

                filter_conds.extend(filters)

            if 'join' in query_plan_node['plan_parameters']:
                # search for join conditions in this node
                try:
                    join = parse_join_condition(query_plan_node['plan_parameters']['join'], tables[0], tables[1],
                                                database_statistics, schema_relationships=schema_relationships)
                except Exception as e:
                    print(query_plan_node)
                    print(database_statistics)
                    raise e

                join_conds.append(join)

    # concatenate tables lists
    tables = [table for sublist in tables for table in sublist]

    if udf_found:
        if 'table_name' in query_plan_node['plan_parameters']:
            tables.append(query_plan_node['plan_parameters']['table_name'])

    return filter_conds, join_conds, tables
