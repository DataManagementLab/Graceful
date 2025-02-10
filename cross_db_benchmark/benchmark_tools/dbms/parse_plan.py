import collections
import re
from typing import Dict

import numpy as np
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.generate_workload import LogicalOperator
from cross_db_benchmark.benchmark_tools.dbms.parse_dd_plan import extract_udf_stats
from cross_db_benchmark.benchmark_tools.dbms.plan_operator import PlanOperator
from cross_db_benchmark.benchmark_tools.dbms.utils import plan_statistics

planning_time_regex = re.compile('planning time: (?P<planning_time>\d+.\d+) ms')
ex_time_regex = re.compile('execution time: (?P<execution_time>\d+.\d+) ms')
init_plan_regex = re.compile("InitPlan \d+ \(returns \$\d\)")
join_columns_regex = re.compile('\w+\.\w+ ?= ?\w+\.\w+')


def create_node(lines_plan_operator, operators_current_level, sql_query: str):
    if len(lines_plan_operator) > 0:
        last_operator = PlanOperator(lines_plan_operator, sql_query=sql_query)
        operators_current_level.append(last_operator)
        lines_plan_operator = []
    return lines_plan_operator


def count_left_whitespaces(a):
    return len(a) - len(a.lstrip(' '))


def parse_recursively(parent, plan, offset, depth, sql_query: str):
    lines_plan_operator = []
    i = offset
    operators_current_level = []
    while i < len(plan):
        # new operator
        if plan[i].strip().startswith('->'):

            # create plan node for previous one
            lines_plan_operator = create_node(lines_plan_operator, operators_current_level, sql_query=sql_query)

            # if plan operator is deeper
            new_depth = count_left_whitespaces(plan[i])
            if new_depth > depth:
                assert len(operators_current_level) > 0, "No parent found at this level"
                i = parse_recursively(operators_current_level[-1], plan, i, new_depth, sql_query=sql_query)

            # one step up in recursion
            elif new_depth < depth:
                break

            # new operator in current depth
            elif new_depth == depth:
                lines_plan_operator.append(plan[i])
                i += 1

        else:
            lines_plan_operator.append(plan[i])
            i += 1

    create_node(lines_plan_operator, operators_current_level, sql_query=sql_query)

    # any node in the recursion
    if parent is not None:
        parent.children = operators_current_level
        return i

    # top node
    else:
        # there should only be one top node
        assert len(operators_current_level) == 1
        return operators_current_level[0]


def parse_plan(analyze_plan_tuples, analyze=True, parse=True, sql_query: str = None):
    assert sql_query is not None

    plan_steps = analyze_plan_tuples
    if isinstance(analyze_plan_tuples[0], tuple) or isinstance(analyze_plan_tuples[0], list):
        plan_steps = [t[0] for t in analyze_plan_tuples]

    # for some reason this is missing in postgres
    # in order to parse this, we add it
    plan_steps[0] = '->  ' + plan_steps[0]

    ex_time = 0
    planning_time = 0
    planning_idx = -1
    if analyze:
        for i, plan_step in enumerate(plan_steps):
            plan_step = plan_step.lower()
            ex_time_match = planning_time_regex.match(plan_step)
            if ex_time_match is not None:
                planning_idx = i
                planning_time = float(ex_time_match.groups()[0])

            ex_time_match = ex_time_regex.match(plan_step)
            if ex_time_match is not None:
                ex_time = float(ex_time_match.groups()[0])

        assert ex_time != 0 and planning_time != 0
        plan_steps = plan_steps[:planning_idx]

    root_operator = None
    if parse:
        root_operator = parse_recursively(None, plan_steps, 0, 0, sql_query=sql_query)

    return root_operator, ex_time, planning_time


def parse_plans(run_stats, min_runtime_ms=100, max_runtime_ms=30000, parse_baseline=False, cap_queries=None,
                parse_join_conds=False, include_zero_card=False, explain_only=False,
                udf_code_dict: Dict[str, str] = None):
    # keep track of column statistics
    column_id_mapping = dict()
    table_id_mapping = dict()
    partial_column_name_mapping = collections.defaultdict(set)

    database_stats = run_stats.database_stats
    # enrich column stats with table sizes
    table_sizes = dict()
    for table_stat in database_stats.table_stats:
        table_sizes[table_stat.relname] = table_stat.reltuples

    for i, column_stat in enumerate(database_stats.column_stats):
        table = column_stat.tablename
        column = column_stat.attname
        column_stat.table_size = table_sizes[table]
        column_id_mapping[(table, column)] = i
        partial_column_name_mapping[column].add(table)

    # similar for table statistics
    for i, table_stat in enumerate(database_stats.table_stats):
        table = table_stat.relname
        table_id_mapping[table] = i

    # parse individual queries
    parsed_plans = []
    avg_runtimes_ms = []
    no_tables = []
    no_filters = []
    op_perc = collections.defaultdict(int)
    timeout_count = 0
    for q in tqdm(run_stats.query_list):

        # either only parse explain part of query or skip entirely
        curr_explain_only = explain_only
        # do not parse timeout queries
        if hasattr(q, 'timeout') and q.timeout:
            timeout_count += 1
            continue

        alias_dict = dict()
        if not curr_explain_only:
            if q.analyze_plans is None:
                continue

            if len(q.analyze_plans) == 0:
                continue

            # subqueries are currently not supported
            analyze_str = ''.join([l[0] for l in q.verbose_plan])
            if 'SubPlan' in analyze_str or 'InitPlan' in analyze_str:
                continue

            # subquery is empty due to logical constraints
            if '->  Result  (cost=0.00..0.00 rows=0' in analyze_str:
                continue

            # check if it just initializes a plan
            if isinstance(q.analyze_plans[0][0], list):
                analyze_plan_string = ''.join(l[0] for l in q.analyze_plans[0])
            else:
                analyze_plan_string = ''.join(q.analyze_plans)
            if init_plan_regex.search(analyze_plan_string) is not None:
                continue

            # compute average execution and planning times
            ex_times = []
            planning_times = []
            for analyze_plan in q.analyze_plans:
                _, ex_time, planning_time = parse_plan(analyze_plan, analyze=True, parse=False, sql_query=q.sql)
                ex_times.append(ex_time)
                planning_times.append(planning_time)
            avg_runtime_ms = sum(ex_times) / len(ex_times)

            # parse the plan as a tree
            analyze_plan, _, _ = parse_plan(q.analyze_plans[0], analyze=True, parse=True, sql_query=q.sql)
            assert len(analyze_plan) > 0, q.analyze_plans[0]

            # parse information contained in operator nodes (different information in verbose and analyze plan)
            analyze_plan.parse_lines_recursively(alias_dict=alias_dict, parse_baseline=parse_baseline,
                                                 parse_join_conds=parse_join_conds)

        # elif timeout:
        #     avg_runtime = float(2 * max_runtime)

        else:
            avg_runtime_ms = 0

        # only explain plan (not executed)
        verbose_plan, _, _ = parse_plan(q.verbose_plan, analyze=False, parse=True, sql_query=q.sql)
        verbose_plan.parse_lines_recursively(alias_dict=alias_dict, parse_baseline=parse_baseline,
                                             parse_join_conds=parse_join_conds)

        if not curr_explain_only:
            # merge the plans with different information
            analyze_plan.merge_recursively(verbose_plan)

        else:
            analyze_plan = verbose_plan

        tables, filter_columns, operators = plan_statistics(analyze_plan)

        try:
            analyze_plan.parse_columns_bottom_up(column_id_mapping, partial_column_name_mapping, table_id_mapping,
                                                 alias_dict=alias_dict)
        except Exception as e:
            print(f'content: {analyze_plan}', flush=True)
            raise e

        analyze_plan.plan_runtime_ms = avg_runtime_ms

        def augment_no_workers(p, top_no_workers=0):
            no_workers = p.plan_parameters.get('workers_planned')
            if no_workers is None:
                no_workers = top_no_workers

            p.plan_parameters['workers_planned'] = top_no_workers

            for c in p.children:
                augment_no_workers(c, top_no_workers=no_workers)

        augment_no_workers(analyze_plan)

        if not curr_explain_only:
            # check if result is None
            if analyze_plan.min_card() == 0 and not include_zero_card:
                continue

            if min_runtime_ms is not None and avg_runtime_ms < min_runtime_ms:
                continue

            if avg_runtime_ms > max_runtime_ms:
                continue

        def det_udf_pos(analyze_plan):
            """
            Helper function to determine the lowest position of a UDF in the plan tree.
            Further, we determine the number of occurrences of a UDF in the plan. E.g., when a UDF is used in a group by
            then it will show up twice in the plan.
            """
            to_visit = [analyze_plan]
            childs = []
            current_level = 0
            lowest_pos = None
            udf_occurences = 0
            udf_name = ""
            remove_plan = False

            while len(to_visit) > 0 or len(childs) > 0:

                # if all nodes at one level are visited, go to childs at the next level
                if len(to_visit) == 0:
                    to_visit = childs
                    childs = []
                    current_level += 1

                current_node = to_visit.pop()
                childs += current_node["children"]

                # Important Note:
                # In the current setting, a UDF can only occur multiple times in output_columns but only
                # ONCE in a filter column. Hence, we do not have to remove higher level udfs in filter columns later on

                if "output_columns" in current_node["plan_parameters"].keys():
                    # get the output columns of the current node
                    out_cols = current_node["plan_parameters"]["output_columns"]

                    for col in out_cols:
                        if col["udf_name"] is not None:
                            udf_occurences += 1
                            lowest_pos = current_level
                            udf_name = col["udf_name"]

                # transfer the UDF params columns from string to id
                if "udf_params" in current_node["plan_parameters"].keys():
                    if current_node["plan_parameters"]["udf_params"] is not None:
                        col_ids = []
                        table_name = current_node["plan_parameters"]["udf_table"]
                        for col in current_node["plan_parameters"]["udf_params"]:
                            if (table_name, col) in column_id_mapping.keys():
                                col_ids.append(column_id_mapping[(table_name, col)])
                            else:
                                raise Exception(
                                    f"Column name {table_name}.{col} not found in column_id_mapping {column_id_mapping.keys()}")
                        current_node["plan_parameters"]["udf_params"] = col_ids

                # transfer table name from string to id
                if "udf_table" in current_node["plan_parameters"].keys():
                    if current_node["plan_parameters"]["udf_table"] is not None:
                        table_name = current_node["plan_parameters"]["udf_table"]
                        if table_name in table_id_mapping.keys():
                            current_node["plan_parameters"]["udf_table"] = table_id_mapping[table_name]
                        else:
                            raise Exception(
                                f"Table name {table_name} not found in table_id_mapping {table_id_mapping.keys()}")

                if "filter_columns" in current_node["plan_parameters"].keys():
                    if current_node["plan_parameters"]["filter_columns"]["udf_name"] is not None:
                        udf_occurences += 1
                        lowest_pos = current_level
                        udf_name = current_node["plan_parameters"]["filter_columns"]["udf_name"]
                    # small addition when there are index conditions and UDFs
                    # this creates again some shitty stuff with AND and this will fuck up the plan batching ...
                    # elif len(current_node["plan_parameters"]["filter_columns"]["children"]) > 0:
                    #     remove_plan = True

            return lowest_pos, udf_occurences, udf_name, remove_plan

        lowest_pos, udf_occurences, udf_name, remove_plan = det_udf_pos(analyze_plan)

        # check whether UDF is importing math / numpy lib
        udf_math_lib_imported, udf_np_lib_imported, udf_num_math_calls, udf_num_np_calls = extract_udf_stats(udf_name,
                                                                                                             udf_code_dict)

        if remove_plan:
            print('Remove plan because of udf')
            continue

        """
        If the udf occurs more than once in output columns of the plan, we only keep the lowest occurence and set the others to None.
        The lowest occurence is the one that has the highest "distance" from the root operator.
        """
        if udf_occurences > 1:
            def traverse_set_to_none(plan_node, lowest_pos):
                to_visit = [analyze_plan]
                childs = []
                current_level = 0

                while len(to_visit) > 0 or len(childs) > 0:
                    # if all nodes at one level are visited, go to childs at the next level
                    if len(to_visit) == 0:
                        to_visit = childs
                        childs = []
                        current_level += 1

                    current_node = to_visit.pop()
                    childs += current_node["children"]
                    if "output_columns" in current_node["plan_parameters"].keys():
                        # get the output columns of the current node
                        out_cols = current_node["plan_parameters"]["output_columns"]

                        if current_level < lowest_pos:
                            new_out_cols = []
                            for col in out_cols:
                                if col["udf_name"] is not None:
                                    col["udf_name"] = None
                                    col["udf_output"] = "False"
                                if col["aggregation"] is not None and col["columns"] is not None:
                                    new_out_cols.append(col)

                            current_node["plan_parameters"]["output_columns"] = new_out_cols

            traverse_set_to_none(analyze_plan, lowest_pos)

        # put the udf_name into the plan_parameters of the top level operator since this is used later during graph creation
        analyze_plan['udf'] = dict(udf_name=udf_name, udf_math_lib_imported=udf_math_lib_imported,
                                   udf_numpy_lib_imported=udf_np_lib_imported, udf_num_math_calls=udf_num_math_calls,
                                   udf_num_np_calls=udf_num_np_calls)

        # collect statistics
        avg_runtimes_ms.append(avg_runtime_ms)
        no_tables.append(len(tables))
        for _, op in filter_columns:
            op_perc[op] += 1
        # log number of filters without counting AND, OR
        ctr = len([fc for fc in filter_columns if fc[0] is not None])
        no_filters.append(ctr)

        analyze_plan.num_tables = len(tables)
        analyze_plan.num_filters = ctr
        analyze_plan.query = q.sql

        parsed_plans.append(analyze_plan)

        if cap_queries is not None and len(parsed_plans) >= cap_queries:
            print(f"Parsed {cap_queries} queries. Stopping parsing.")
            break

        if parse_baseline:
            def list_columns(n, columns):
                if n['operator'] != str(LogicalOperator.AND) and n['operator'] != str(LogicalOperator.OR):
                    columns.append((n['column'], n['operator'], n['literal']))
                for c in n['children']:
                    list_columns(c, columns)

            def list_col_rec(n, columns):
                filter_column = n.plan_parameters.get('filter_columns')
                if filter_column is not None:
                    list_columns(filter_column, columns)
                for c in n.children:
                    list_col_rec(c, columns)

            columns = []
            list_col_rec(analyze_plan, columns)
            if ' JOIN ' in q.sql:
                if ' WHERE ' not in q.sql:
                    exp_no_filter = 0
                else:
                    exp_no_filter = q.sql.split(' WHERE ')[1].count(' AND ') + 1

                if not exp_no_filter <= len(columns):
                    print(f"Warning: did not find enough filters exp_no_filter: {exp_no_filter} ({q.sql}), "
                          f"columns: {columns}")

    # statistics in seconds
    assert no_tables is not None

    if len(no_tables) > 0:
        mean_t = f'{np.mean(no_tables):.1f}'
        median_t = f'{np.median(no_tables):.1f}'
        max_t = f'{np.max(no_tables):.1f}'
        mean_f = f'{np.mean(no_filters):.1f}'
        median_f = f'{np.median(no_filters):.1f}'
        max_f = f'{np.max(no_filters):.1f}'
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
    print("Operators statistics (appear in x% of queries)")
    for op, op_count in op_perc.items():
        print(f"\t{str(op)}: {op_count / len(avg_runtimes_ms) * 100:.0f}%")
    print(f"Runtime statistics: "
          f"\n\tmedian: {median_r}s"
          f"\n\tmax: {max_r}s"
          f"\n\tmean: {mean_r}s")
    print(f"Parsed {len(parsed_plans)} plans ({len(run_stats.query_list) - len(parsed_plans)} had zero-cardinalities "
          f"or were too fast).")

    parsed_runs = dict(parsed_plans=parsed_plans, database_stats=database_stats,
                       run_kwargs=run_stats.run_kwargs)

    stats = dict(
        runtimes=str(avg_runtimes_ms),
        no_tables=str(no_tables),
        no_filters=str(no_filters)
    )

    return parsed_runs, stats


def normalize_join_condition(p_join_str):
    join_conds = p_join_str.split('AND')
    join_conds = [normalize_single_join_condition(jc.strip()) for jc in join_conds]
    join_conds = sorted(join_conds)
    join_conds = ' AND '.join(join_conds)
    return join_conds


def normalize_single_join_condition(p_join_str):
    join_cond = p_join_str.split('=')
    assert len(join_cond) == 2
    for i in [0, 1]:
        join_cond[i] = join_cond[i].strip()
    join_cond = sorted(join_cond)
    join_cond = f'{join_cond[0]} = {join_cond[1]}'
    return join_cond
