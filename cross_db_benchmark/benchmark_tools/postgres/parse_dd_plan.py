import collections
import json
import math
import re
from typing import Dict, Tuple, List

import numpy as np
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.generate_workload import Aggregator, ExtendedAggregator
from cross_db_benchmark.benchmark_tools.postgres.parse_filter import parse_filter, PredicateNode, \
    NodeOpNotRecognizedException, FilterParsingError
from cross_db_benchmark.benchmark_tools.postgres.utils import plan_statistics, remove_cast_nesting, \
    extract_udf_params_from_str


def extract_udf_stats(udf_name: str, udf_code_dict):
    udf_math_lib_imported = False
    udf_np_lib_imported = False
    udf_num_math_calls = 0
    udf_num_np_calls = 0
    udf_num_branches = 0
    udf_num_loops = 0

    if udf_name is not None and udf_name != '':
        assert udf_code_dict is not None

        # get udf information
        udf_code = udf_code_dict[udf_name]
        assert isinstance(udf_code, list), f'{type(udf_code)} \n {udf_code}'

        # iterate over code lines of udf and check whether math / numpy is used
        for line in udf_code:
            if 'import math' in line:
                udf_math_lib_imported = True
            if 'import numpy' in line:
                udf_np_lib_imported = True

            udf_num_math_calls += line.count('math.')
            udf_num_np_calls += line.count('np.')
            udf_num_np_calls += line.count('numpy.')

            if line.strip().startswith('for'):
                udf_num_loops += 1
            if line.strip().startswith('while'):
                udf_num_loops += 1
            if line.strip().startswith('if'):
                udf_num_branches += 1

        if udf_num_math_calls > 0:
            udf_math_lib_imported = True
        if udf_num_np_calls > 0:
            udf_np_lib_imported = True

    return udf_math_lib_imported, udf_np_lib_imported, udf_num_math_calls, udf_num_np_calls, udf_num_branches, udf_num_loops


def parse_dd_plans(run_stats, min_runtime_ms: int, max_runtime_ms: int, parse_baseline: bool, cap_queries: int = None,
                   parse_join_conds: bool = True, include_zero_card=False, explain_only=False,
                   udf_code_dict: Dict[str, str] = None, prune_plans: bool = False):
    """
    Parse the plans from the run_stats and return them as a dictionary.
    :param run_stats: the run stats to parse
    :param min_runtime: the minimum runtime of a query to be considered (ms)
    :param max_runtime: the maximum runtime of a query to be considered (ms)
    :param cap_queries: number of queries to parse
    :param parse_join_conds: parse the join conditions
    :param include_zero_card: include zero cardinality queries
    """

    assert parse_join_conds
    assert not explain_only

    # map column name (table, column) -> column id
    column_id_mapping = dict()

    # map table name (table) -> table id
    table_id_mapping = dict()

    # map column name (column) -> set of tables
    partial_column_name_mapping = collections.defaultdict(set)

    database_stats = run_stats.database_stats

    # enrich column stats with table sizes & store table id mapping
    table_sizes = dict()
    for i, table_stats in enumerate(database_stats.table_stats):
        table_sizes[table_stats.table_name] = table_stats.estimated_size
        table_id_mapping[table_stats.table_name] = i

    for i, column_stat in enumerate(database_stats.column_stats):
        table = column_stat.table_name
        column = column_stat.column_name
        column_stat.table_size = table_sizes[table]
        column_id_mapping[(table, column)] = i
        partial_column_name_mapping[column].add(table)

    # parse the plans
    parsed_plans = []
    avg_runtimes_ms = []
    no_tables_list = []
    no_filters_list = []
    op_perc = collections.defaultdict(int)
    timeout_count = 0
    zero_card_count = 0
    no_plans_count = 0
    pruned_plans_count = 0
    subqueries_count = 0
    min_time_count = 0
    max_time_count = 0
    col_arith_count = 0
    col_not_matching_count = 0
    filter_op_not_recognized_count = 0
    filter_parsing_error_count = 0

    for q in tqdm(run_stats.query_list):
        if cap_queries is not None and len(parsed_plans) >= cap_queries:
            print(f"Reached cap of {cap_queries} queries. Stop parsing.")
            break

        # do not parse timeout queries
        if hasattr(q, 'timeout') and q.timeout:
            timeout_count += 1
            continue

        # do not parse queries with no plans
        if q.analyze_plans is None:
            no_plans_count += 1
            continue

        # do not parse queries with no plans
        if len(q.analyze_plans) == 0:
            no_plans_count += 1
            continue

        sql = q.sql

        # compute average execution times
        ex_times = []
        for analyze_plan in q.analyze_plans:
            ex_times.append(float(analyze_plan.result) * 1000)  # in ms

        min_ex_times_ms = min(ex_times)
        max_ex_times_ms = max(ex_times)
        avg_runtime_ms = sum(ex_times) / len(ex_times)

        if min_ex_times_ms < min_runtime_ms:
            # skip queries with runtime outside the range
            min_time_count += 1
            continue

        if max_ex_times_ms > max_runtime_ms:
            # skip queries with runtime outside the range
            max_time_count += 1
            continue

        if prune_plans:
            t0 = ex_times[0]
            tlast = ex_times[-1]
            if t0 < tlast and tlast / t0 > 1.05:
                # skip queries with runtime outside the range
                pruned_plans_count += 1
                continue

            avg_runtime_ms = tlast

        try:
            parsed_plan = parse_plan_recursively(q.analyze_plans[0], include_zero_card=include_zero_card,
                                                 table_id_mapping=table_id_mapping, column_id_mapping=column_id_mapping,
                                                 sql_query=sql)
        except ZeroCardFoundError as e:
            # skip queries with zero cardinality
            zero_card_count += 1
            continue
        except ColArithInPlanFoundError as e:
            col_arith_count += 1
            continue
        except ColNotMatchingError as e:
            # found a column which is not matching to the tables/columns returned by its children
            col_not_matching_count += 1
            continue
        except NodeOpNotRecognizedException as e:
            # filter could not be parsed
            filter_op_not_recognized_count += 1
            continue
        except FilterParsingError as e:
            # filter could not be parsed
            filter_parsing_error_count += 1
            continue
        except Exception as e:
            print(q.analyze_plans[0])
            raise e

        udf_names = recursively_get(parsed_plan, 'udf_name')
        assert len(udf_names) <= 1, f"UDF names should not exceed length 1, but has length {len(udf_names)}"
        if len(udf_names) > 0:
            udf_name = udf_names[0]
        else:
            udf_name = None

        if 'func_' in sql:
            assert udf_name is not None, f"UDF name should not be None if func_ is in sql: {sql}\n{parsed_plan}\n{q.analyze_plans[0]}"

        # check whether UDF is importing math / numpy lib
        udf_math_lib_imported, udf_np_lib_imported, udf_num_math_calls, udf_num_np_calls, udf_num_branches, udf_num_loops = extract_udf_stats(
            udf_name,
            udf_code_dict)
        if udf_name is not None:
            # retrieve udf pos in query
            udf_pos_in_query_list = get_udf_pos_in_query(parsed_plan)

            # parse to select / filter / group_by
            assert len(
                udf_pos_in_query_list) == 1, f'only 1 udf supported at the moment, received: {udf_pos_in_query_list}'
            if udf_pos_in_query_list[0] == 'FILTER':
                udf_pos_in_query = 'filter'
            elif udf_pos_in_query_list[0] in ['PROJECTION', 'UNGROUPED_AGGREGATE']:
                udf_pos_in_query = 'select'
            else:
                raise Exception(f'UDF position in query not recognized: {udf_pos_in_query_list[0]} /n {parsed_plan}')

            # put the udf_name into the plan_parameters of the top level operator since this is used later during graph creation
            parsed_plan['udf'] = dict(
                udf_name=udf_name, udf_math_lib_imported=udf_math_lib_imported,
                udf_numpy_lib_imported=udf_np_lib_imported,
                udf_num_math_calls=udf_num_math_calls,
                udf_num_np_calls=udf_num_np_calls,
                udf_pos_in_query=udf_pos_in_query,
                udf_num_branches=udf_num_branches,
                udf_num_loops=udf_num_loops
            )

        parsed_plan["query"] = q.sql

        # extract num tables and num filters
        tables, filter_columns, operators = plan_statistics(parsed_plan)
        # log number of filters without counting AND, OR
        no_filters_ctr = len([fc for fc in filter_columns if fc[0] is not None])

        # log operators
        for _, op in filter_columns:
            op_perc[op] += 1

        parsed_plan['plan_runtime_ms'] = avg_runtime_ms

        parsed_plan['num_tables'] = len(tables)
        parsed_plan['num_filters'] = no_filters_ctr

        # save the parsed plan
        parsed_plans.append(parsed_plan)
        avg_runtimes_ms.append(avg_runtime_ms)

        no_tables_list.append(len(tables))
        no_filters_list.append(no_filters_ctr)

    # statistics in seconds
    assert no_tables_list is not None

    if len(no_tables_list) > 0:
        mean_t = f'{np.mean(no_tables_list):.1f}'
        median_t = f'{np.median(no_tables_list):.1f}'
        max_t = f'{np.max(no_tables_list):.1f}'
        mean_f = f'{np.mean(no_filters_list):.1f}'
        median_f = f'{np.median(no_filters_list):.1f}'
        max_f = f'{np.max(no_filters_list):.1f}'
        mean_r = f'{np.mean(avg_runtimes_ms) / 1000:.2f}'
        median_r = f'{np.median(avg_runtimes_ms) / 1000:.2f}'
        max_r = f'{np.max(avg_runtimes_ms) / 1000:.2f}'
    else:
        mean_t = None
        median_t = None
        max_t = None
        mean_f = None
        median_f = None
        max_f = None
        mean_r = None
        median_r = None
        max_r = None

    print(f"Table statistics: "
          f"\n\tmean: {mean_t}"
          f"\n\tmedian: {median_t}"
          f"\n\tmax: {max_t}")
    print(f"Filter statistics: "
          f"\n\tmean: {mean_f}"
          f"\n\tmedian: {median_f}"
          f"\n\tmax: {max_f}")
    print("Number of timeouts: ", timeout_count)
    print("Number of queries with no plans: ", no_plans_count)
    print("Number of zero-card queries: ", zero_card_count)
    print("Number of queries with subqueries: ", subqueries_count)
    print(f"Number of queries with runtime lower than min time ({min_runtime_ms}ms): ", min_time_count)
    print(f"Number of queries with runtime higher than max time ({max_runtime_ms}ms): ", max_time_count)
    print(f"Number of queries pruned because variance too high between runs: ", pruned_plans_count)
    print(f'Number of queries with arithmetic ops between columns: {col_arith_count}')
    print(f'Number of queries with columns not matching to tables: {col_not_matching_count}')
    print(f'Number of queries with filters op not recognized: {filter_op_not_recognized_count}')
    print(f'Number of queries with filter parsing error: {filter_parsing_error_count}')
    print("Operators statistics (appear in x% of queries)")
    for op, op_count in op_perc.items():
        print(f"\t{str(op)}: {op_count / len(avg_runtimes_ms) * 100:.0f}%")
    print(f"Runtime statistics: "
          f"\n\tmedian: {median_r}s"
          f"\n\tmax: {max_r}s"
          f"\n\tmean: {mean_r}s")
    print(
        f"Parsed {len(parsed_plans)} plans ({len(run_stats.query_list) - len(parsed_plans)} had no plans, "
        f"zero-cardinalities or were too fast).")

    parsed_runs = dict(parsed_plans=parsed_plans, database_stats=database_stats, run_kwargs=run_stats.run_kwargs, )
    stats = dict(runtimes=str(avg_runtimes_ms), no_tables=str(no_tables_list), no_filters=str(no_filters_list))

    return parsed_runs, stats


class ZeroCardFoundError(Exception):
    def __init__(self, message):
        super().__init__(message)


def _parse_out_col_list(output_columns: str) -> List[str]:
    # split string by comma, but ignore commas in parenthesis
    col_list = []

    cur_text = ''
    opened_parenthesis = 0

    for c in output_columns:
        if (c == ',' or c == '\n') and opened_parenthesis == 0:
            if len(cur_text) > 0:
                col_list.append(cur_text.strip())
            cur_text = ''
        else:
            cur_text += c
            if c == '(':
                opened_parenthesis += 1
            elif c == ')':
                opened_parenthesis -= 1

    if len(cur_text) > 0:
        col_list.append(cur_text.strip())

    return col_list


class ColArithInPlanFoundError(Exception):
    # raised e.g. when two columns are added in the output. This is not yet supported in the parser.
    def __init__(self, message):
        super().__init__(message)


def parse_output_columns(output_columns: str, column_id_mapping: Dict[Tuple[str, str], int],
                         table_id_mapping: Dict[str, int],
                         table_name: str = None, child_tables: List[str] = None) -> Tuple[
    List[Dict], List[int], List[List[int]]]:
    try:
        # if ' + ' in output_columns or ' - ' in output_columns or ' * ' in output_columns or ' / ' in output_columns:
        #     raise ColArithInPlanFoundError(f"Column arithmetic found in plan: {output_columns}")

        output_columns_list = _parse_out_col_list(output_columns)

        output_column_json = []
        udf_table_names = []
        udf_params_lists = []

        for col in output_columns_list:
            col = remove_cast_nesting(col)[0]

            if col.startswith('func_'):
                # parse udf function
                udf_name = col.split('(')[0]
                udf_args = col.removeprefix(f'{udf_name}(').removesuffix(')')

                # parse udf args
                udf_args_parsed = _parse_out_col_list(udf_args)

                # list storing candidates for table name of each argument (tables can be ambiguous)
                arg_table_candidates = []
                # list storing the column name of each argument
                arg_col_list = []

                for arg in udf_args_parsed:
                    original_arg = arg
                    arg = arg.strip()
                    # cast should be stripped beforehand - ignored for now
                    assert 'CAST(' not in arg, arg
                    if '.' in arg:
                        assert len(arg.split('.')) == 2, arg
                        arg_table, arg_col = arg.split('.')
                    else:
                        if table_name is not None:
                            arg_table = table_name
                        else:
                            arg_table = _reconstruct_table_name_from_col_id(arg, column_id_mapping, child_tables,
                                                                            return_ambg_tables=True)

                        arg_col = arg
                    assert arg_col != '', f'arg: {arg}, col: {col}, original arg: {original_arg}\n{udf_args_parsed}'

                    arg_table_candidates.append(arg_table)
                    arg_col_list.append(arg_col)

                # check that there is one table candidate which is contained in every argument (i.e. one table having all the referenced columns)
                table_ctr_dict = collections.defaultdict(int)
                for arg_table in arg_table_candidates:
                    if isinstance(arg_table, list):
                        for table in arg_table:
                            table_ctr_dict[table] += 1
                    else:
                        table_ctr_dict[arg_table] += 1
                matching_tables = [k for k, v in table_ctr_dict.items() if v == len(arg_table_candidates)]
                assert len(
                    matching_tables) >= 1, f'No matching table found for UDF {udf_name} with args {udf_args_parsed} (table candidates: {arg_table_candidates})'
                tables = set(matching_tables[:1])

                # create args list
                args = []
                for arg_col in arg_col_list:
                    args.append(column_id_mapping[(matching_tables[0], arg_col)])

                assert len(
                    tables) == 1, f"UDF args {udf_args_parsed} contain columns from multiple tables: {tables}, {matching_tables}"

                udf_table_names.append(table_id_mapping[list(tables)[0]])
                udf_params_lists.append(args)

                output_column_json.append(
                    dict(aggregation=str(None), columns=None, udf_name=udf_name, udf_output="True", child_ref=False))
            else:
                agg = None
                child_ref = False

                if 'count(*)' in col or 'count_star()' in col:
                    agg = Aggregator.COUNT
                    column_ids = []
                else:
                    if col.startswith('sum_no_overflow('):
                        # replace sum_no_overflow with sum
                        col = f'sum({col.removeprefix("sum_no_overflow(")}'

                    # check whether aggregation is used
                    for curr_agg in list(Aggregator) + list(ExtendedAggregator):
                        if col.startswith(f'{str(curr_agg).lower()}('):
                            agg = curr_agg
                            col = col.removeprefix(str(curr_agg).lower() + '(').removesuffix(')').strip('()')
                            break

                    # when an aggregation is performed on the sum of two columns, it will be split into two operations:
                    # 1. (colA + colB) - projection node (child)
                    # 2. sum(#0) - ungrouped aggregate node (parent)
                    if col.startswith('(') and col.endswith(')'):
                        col = col[1:-1]

                    # check whether column arithmetic is used, e.g. colA + colB
                    col_list = None
                    arith_str_list = [' + ', ' - ', ' * ', ' / ']
                    for arith_str in arith_str_list:
                        if arith_str in col:
                            col_list = col.split(arith_str)
                            break

                    if col_list is None:
                        col_list = [col]
                    else:
                        assert len(col_list) >= 2, f'{col} \ {col_list}'

                    column_ids = []

                    for parsed_col in col_list:
                        if parsed_col.startswith('(') and parsed_col.endswith(')'):
                            parsed_col = parsed_col[1:-1]
                        if parsed_col.startswith('CAST('):
                            # remove cast prefix and suffix
                            parsed_col = parsed_col.removeprefix('CAST(').removesuffix(')')
                            parsed_col = parsed_col.split(' AS ')[0]
                            assert parsed_col != '', parsed_col

                        # no udf
                        if parsed_col.startswith('#'):
                            # column is refererring to output column for child node
                            # this happens when an aggregation is performed on the sum of two columns
                            column_ids.append(parsed_col)
                            child_ref = True
                        elif '.' in parsed_col:
                            assert len(parsed_col.split('.')) == 2, parsed_col
                            table, parsed_col = parsed_col.split('.')
                            column_ids.append(column_id_mapping[(table, parsed_col)])
                        else:
                            if table_name is not None:
                                table = table_name
                            else:
                                try:
                                    table = _reconstruct_table_name_from_col_id(parsed_col, column_id_mapping,
                                                                                child_tables, ignore_ambg_tables=True)
                                except Exception as e:
                                    print(col)
                                    raise e
                            column_ids.append(column_id_mapping[(table, parsed_col)])

                output_column_json.append(
                    dict(aggregation=str(agg), columns=column_ids, udf_name=None, udf_output="False",
                         child_ref=child_ref))

        return output_column_json, udf_table_names, udf_params_lists
    except ColArithInPlanFoundError as e:
        # forward error but do not capture here
        raise e
    except Exception as e:
        print(f'Scan table: {table_name}')
        print(output_columns)
        print(column_id_mapping)
        raise e


class ColNotMatchingError(Exception):
    def __init__(self, message):
        super().__init__(message)


def _reconstruct_table_name_from_col_id(col: str, column_id_mapping: Dict[Tuple[str, str], int],
                                        table_candidates: List[str], return_ambg_tables: bool = False,
                                        ignore_ambg_tables: bool = False):
    # try to get table name for column id dict
    matching_table_names = [k for k, v in column_id_mapping.keys() if v == col]

    if len(matching_table_names) == 1:
        if not matching_table_names[0] in table_candidates:
            raise ColNotMatchingError(
                f"Table {matching_table_names[0]} not in table candidates {table_candidates}\n{col}\n{column_id_mapping}")
        return matching_table_names[0]
    elif len(matching_table_names) == 0:
        raise KeyError(f"Unknown column name: {col}")
    else:
        # Column is ambiguous
        # Try to find the table name in the table candidates
        candidates = []
        for table in table_candidates:
            if (table, col) in column_id_mapping:
                candidates.append(table)
        assert len(candidates) > 0
        if len(candidates) == 1:
            return candidates[0]
        else:
            if return_ambg_tables:
                return candidates
            elif ignore_ambg_tables:
                return candidates[0]
            else:
                raise KeyError(f"Ambiguous column name: {col} (candidates: {candidates})")


def translate_columns_recursively(filter: PredicateNode, column_id_mapping, child_tables: List[str] = None):
    if filter.column is not None:
        if len(filter.column) == 1:
            # find out what the table name is
            try:
                table_name = _reconstruct_table_name_from_col_id(filter.column[0], column_id_mapping, child_tables,
                                                                 return_ambg_tables=True)
            except KeyError as e:
                print(filter)
                raise e
            if isinstance(table_name, list):
                # ambiguous column name, pick random column
                table_name = table_name[0]

            # get id of the column
            filter.column = column_id_mapping[(table_name, filter.column[0])]
        else:
            filter.column = column_id_mapping[filter.column]
    for child in filter.children:
        translate_columns_recursively(child, column_id_mapping=column_id_mapping, child_tables=child_tables)


def parse_plan_recursively(plan, include_zero_card: bool, table_id_mapping: Dict[str, int], sql_query: str,
                           column_id_mapping: Dict[Tuple[str, str], int], translate_table_col_names_to_id: bool = True):
    """
    Parse the plan recursively.
    :param plan: the plan to parse
    :return: the parsed plan
    """
    if plan.name in ['EXPLAIN_ANALYZE', 'Query', 'RESULT_COLLECTOR']:
        # skip this node
        return parse_plan_recursively(plan.children[0], include_zero_card=include_zero_card,
                                      table_id_mapping=table_id_mapping, column_id_mapping=column_id_mapping,
                                      translate_table_col_names_to_id=translate_table_col_names_to_id,
                                      sql_query=sql_query)

    parsed_plan = dict()
    plan_parameters = dict()

    # extract information
    plan_parameters['op_name'] = plan.name.strip()

    # raise exception if cardinality is zero
    if plan.cardinality == 0 and not include_zero_card:
        raise ZeroCardFoundError(f"Cardinality of plan is zero: {plan}")

    plan_parameters['act_card'] = plan.cardinality

    extra_info = plan.extra_info

    # extract estimated card from extra info
    ec_regex = r'EC: ([0-9]+)'
    match = re.search(ec_regex, extra_info)
    if match is not None:
        plan_parameters['est_card'] = int(match.groups()[0])
    else:
        assert 'EC:' not in extra_info, json.dumps(extra_info)
        plan_parameters['est_card'] = -1

    # extract join condition
    join_regex = r'INNER\n([^\n]+)'
    match = re.search(join_regex, extra_info)
    if 'INNER' in extra_info and match is None:
        print(extra_info)
        raise Exception(f"INNER in extra info: {extra_info}")
    if match is not None:
        plan_parameters['join'] = match.group(0)

    is_scan = False
    if 'SCAN' in plan.name:
        is_scan = True
        # extract output columns and table name
        extra_info_splitted = extra_info.split('[INFOSEPARATOR]')
        assert len(extra_info_splitted) > 2, f"Extra info {extra_info} does not contain enough information"
        table_name = extra_info_splitted[0]
        table_name = table_name.strip('\n')
        plan_parameters['table_name'] = table_name
        try:
            if not translate_table_col_names_to_id:
                plan_parameters['table'] = table_name
            else:
                plan_parameters['table'] = table_id_mapping[table_name]
        except KeyError:
            raise KeyError(f"Table {table_name} not found in table_id_mapping: {table_id_mapping}")

        output_columns = extra_info_splitted[1]

        if translate_table_col_names_to_id:
            output_columns, udf_table_list, udf_params_lists = parse_output_columns(output_columns, column_id_mapping,
                                                                                    table_id_mapping=table_id_mapping,
                                                                                    table_name=table_name,
                                                                                    child_tables=None, )
            plan_parameters['output_columns'] = output_columns

            # supports only one udf at the moment
            assert len(
                udf_table_list) <= 1, f"UDF table list should not exceed length 1, but has length {len(udf_table_list)}"
            assert len(
                udf_params_lists) <= 1, f"UDF params list should not exceed length 1, but has length {len(udf_params_lists)}"
            if len(udf_params_lists) == 1:
                assert len(udf_params_lists[0]) > 0, udf_params_lists

                plan_parameters['udf_table'] = udf_table_list[0]
                plan_parameters['udf_params'] = udf_params_lists[0]

        # extract filter condition
        if extra_info_splitted[2].strip().startswith('Filters: '):
            # filter_regex = r'Filters: ([^\n]+)'
            # match = re.search(filter_regex, extra_info)
            # assert 'Filters: ' not in extra_info or match is not None, extra_info
            # if match is not None:
            #     filter_str = match.group(1)
            #     filter = parse_filter(f'({filter_str})', duckdb=True)
            filter_str = extra_info_splitted[2].strip().removeprefix('Filters: ')
            filter_str = ' AND '.join(filter_str.split('\n'))

            filter = parse_filter(f'( {filter_str} )', duckdb=True, sql=sql_query)
            if translate_table_col_names_to_id:
                translate_columns_recursively(filter, column_id_mapping=column_id_mapping, child_tables=[table_name])
            plan_parameters['filter_columns'] = filter

            if 'func_' in filter_str:
                params = extract_udf_params_from_str(filter_str)
                plan_parameters['udf_params'] = [column_id_mapping[(table_name, param)] for param in params]
                plan_parameters['udf_table'] = table_id_mapping[table_name]

    # parse the children
    if not is_scan:
        children = [
            parse_plan_recursively(child, include_zero_card=include_zero_card, table_id_mapping=table_id_mapping,
                                   column_id_mapping=column_id_mapping,
                                   translate_table_col_names_to_id=translate_table_col_names_to_id, sql_query=sql_query)
            for child in
            plan.children]
    else:
        children = []

    child_tables = recursively_get(children, 'table_name')

    if plan.name == 'FILTER':
        extra_info_splitted = extra_info.split('[INFOSEPARATOR]')
        assert len(extra_info_splitted) == 2, f"Extra info {extra_info} does not contain enough information"
        filter_str = extra_info_splitted[0].strip().strip('\n')
        filter_str = ' AND '.join(filter_str.split('\n'))
        filter = parse_filter(f'({filter_str})', duckdb=True, sql=sql_query)

        # translate filter columns to column ids
        if translate_table_col_names_to_id:
            translate_columns_recursively(filter, column_id_mapping=column_id_mapping, child_tables=child_tables)
        plan_parameters['filter_columns'] = filter

        if 'func_' in filter_str and translate_table_col_names_to_id:
            params = extract_udf_params_from_str(filter_str)
            table_col_ids_dict = collections.defaultdict(list)
            for param in params:
                table_name = _reconstruct_table_name_from_col_id(param, column_id_mapping, child_tables,
                                                                 return_ambg_tables=True)
                if isinstance(table_name, list):
                    # multiple possible table names found for this candidate
                    for table in table_name:
                        table_col_ids_dict[table].append(column_id_mapping[(table, param)])
                else:
                    table_col_ids_dict[table_name].append(column_id_mapping[(table_name, param)])

            # select the table name where all columns match
            found = False
            col_ids = None
            table_name = None
            assert len(table_col_ids_dict) > 0, f"Table col ids dict is empty: {table_col_ids_dict}"
            for table_name, col_ids in table_col_ids_dict.items():
                if len(col_ids) == len(params):
                    found = True
                    break

            assert found, f"Could not find table name where all columns match: {table_col_ids_dict}"

            plan_parameters['udf_params'] = col_ids
            plan_parameters['udf_table'] = table_id_mapping[table_name]

    # parse the projection operation, sometimes the table name is omitted, so we extract if from the child nodes
    # This needs to be considered:
    # {
    #     "name": "UNGROUPED_AGGREGATE",
    #     "timing": 0.005341,
    #     "cardinality": 1,
    #     "extra_info": "avg(#0)\nsum(#1)",
    #     "timings": [],
    #     "children": [
    #         {
    #             "name": "PROJECTION",
    #             "timing": 0.008041,
    #             "cardinality": 509371,
    #             "extra_info": "(lfta + lteam)\n(lstl + lor)\n",
    #             "timings": [],
    #             "children": [
    #                 {
    if plan.name in ['PROJECTION', 'UNGROUPED_AGGREGATE']:
        # extract output columns
        extra_info_splitted = extra_info.split('[INFOSEPARATOR]')
        assert len(extra_info_splitted) > 0, f"Extra info {extra_info} does not contain enough information"
        output_columns_raw = extra_info_splitted[0]

        if plan.name == 'PROJECTION':
            output_cols = output_columns_raw.strip().split('\n')
            real_col_found = False
            for col in output_cols:
                if col.startswith('#') or col.strip() == '':
                    continue
                else:
                    real_col_found = True
                    break

            if not real_col_found:
                # this is a projection node which only passes through the child output
                # skip this node and return child node
                assert len(children) == 1, children
                return children[0]

        if translate_table_col_names_to_id:
            output_columns, udf_table_list, udf_params_lists = parse_output_columns(output_columns_raw,
                                                                                    column_id_mapping,
                                                                                    table_id_mapping=table_id_mapping,
                                                                                    table_name=child_tables[0] if len(
                                                                                        child_tables) == 1 else None,
                                                                                    child_tables=child_tables)
            assert len(output_columns) > 0, f"Output columns {output_columns} is empty (raw: {output_columns_raw})"

            skip_child = False  # skip child if output columns are pulled up
            child_outcols_pulled_up = []  # list of output columns which are pulled up from child

            # check whether output column is referring to a child
            # and in this case pull up the output column info from the child
            for i in range(len(output_columns)):
                if output_columns[i]['child_ref']:
                    if len(output_columns[i]['columns']) > 1:
                        # reduce to one column
                        # TODO in future we might want to support multiple columns - for now ignore further aggr.
                        reduced_c = [c for c in output_columns[i]['columns'] if not isinstance(c, int) and '#' in c]
                        assert len(reduced_c) > 0, output_columns[i]
                        output_columns[i]['columns'] = reduced_c[:1]  # take only one entry

                    assert len(output_columns[i]['columns']) == 1, output_columns[i]
                    child_ref = output_columns[i]['columns'][0]  # a reference with type of '#0' / '#1' / ...
                    child_ref_int = int(child_ref[1:])

                    # merge output column info which child
                    assert len(children) == 1, children

                    if child_ref_int > len(children[0]['plan_parameters']['output_columns']) - 1:
                        # TODO in future we might want to support multiple columns - for now ignore further aggr.
                        child_ref_int = 0

                    referred_child_out_col = children[0]['plan_parameters']['output_columns'][child_ref_int]

                    # check consistency of child output column
                    assert referred_child_out_col['aggregation'] == 'None', referred_child_out_col
                    assert output_columns[i]['udf_name'] is None, output_columns
                    assert output_columns[i]['udf_output'] == 'False', output_columns
                    assert referred_child_out_col['child_ref'] == False, referred_child_out_col
                    if referred_child_out_col['columns'] is None:
                        assert referred_child_out_col['udf_name'] is not None, referred_child_out_col
                    else:
                        assert referred_child_out_col['columns'] is not None and len(
                            referred_child_out_col['columns']) > 0, referred_child_out_col

                    # merge output column info
                    output_columns[i]['columns'] = referred_child_out_col['columns']
                    output_columns[i]['child_ref'] = False
                    output_columns[i]['udf_name'] = referred_child_out_col['udf_name']
                    output_columns[i]['udf_output'] = referred_child_out_col['udf_output']

                    skip_child = True
                    child_outcols_pulled_up.append(child_ref_int)

            if skip_child:
                # check whether child output columns are all pulled up
                for i in range(len(children[0]['plan_parameters']['output_columns'])):
                    if i not in child_outcols_pulled_up:
                        # child output column is not pulled up, add it to this node
                        output_columns.append(children[0]['plan_parameters']['output_columns'][i])

                # copy over udf info from child
                if 'udf_table' in children[0]['plan_parameters']:
                    plan_parameters['udf_table'] = children[0]['plan_parameters']['udf_table']
                    plan_parameters['udf_params'] = children[0]['plan_parameters']['udf_params']

                # skip child
                children = children[0]['children']

            plan_parameters['output_columns'] = output_columns

            # supports only one udf at the moment
            assert len(
                udf_table_list) <= 1, f"UDF table list should not exceed length 1, but has length {len(udf_table_list)}"
            assert len(
                udf_params_lists) <= 1, f"UDF params list should not exceed length 1, but has length {len(udf_params_lists)}"
            if len(udf_params_lists) == 1:
                assert len(udf_params_lists[0]) > 0, udf_params_lists

                plan_parameters['udf_table'] = udf_table_list[0]
                plan_parameters['udf_params'] = udf_params_lists[0]

    # calculate children cardinality
    def child_prod(children: List, feature_name: str, default=1):
        child_feat = [c['plan_parameters'][feature_name] for c in children
                      if feature_name in c['plan_parameters'] and c['plan_parameters'][feature_name] is not None]
        if len(child_feat) == 0:
            return default
        return math.prod(child_feat)

    plan_parameters['act_children_card'] = child_prod(children, 'act_card')
    plan_parameters['est_children_card'] = child_prod(children, 'est_card')

    if plan_parameters['est_card'] == -1:
        # estimated cardinality not explicitly stated in raw plan. Try to infer
        if plan_parameters['op_name'] == 'PROJECTION':
            # cardinality of projection is the same as the child
            assert len(children) == 1, children
            plan_parameters['est_card'] = plan_parameters['est_children_card']
        elif plan_parameters['op_name'] == 'UNGROUPED_AGGREGATE':
            # one row will be returned
            plan_parameters['est_card'] = 1

    parsed_plan['plan_parameters'] = plan_parameters
    parsed_plan['children'] = children

    return parsed_plan


def recursively_get(plan, attribute):
    # if plan is a list, recursively get the attribute for each element
    if isinstance(plan, list):
        res = []
        for p in plan:
            res.extend(recursively_get(p, attribute))
        return res
    else:
        res = []
        if 'plan_parameters' in plan and attribute in plan['plan_parameters']:
            res.append(plan['plan_parameters'][attribute])

        if attribute == 'udf_name':
            if 'plan_parameters' in plan and 'output_columns' in plan['plan_parameters']:
                for out_col in plan['plan_parameters']['output_columns']:
                    if 'udf_name' in out_col and out_col['udf_name'] is not None:
                        res.append(out_col['udf_name'])
            if 'plan_parameters' in plan and 'filter_columns' in plan['plan_parameters']:
                res.extend(plan['plan_parameters']['filter_columns'].get_udf_names())

        for child in plan['children']:
            res.extend(recursively_get(child, attribute))

        return res


def get_udf_pos_in_query(plan):
    res = []
    if 'plan_parameters' in plan and 'udf_table' in plan['plan_parameters']:
        res.append(plan['plan_parameters']['op_name'])

    for child in plan['children']:
        res.extend(get_udf_pos_in_query(child))

    return res
