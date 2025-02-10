import json
import traceback
from types import SimpleNamespace

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.dbms.parse_dd_plan import parse_plan_recursively, ZeroCardFoundError
from cross_db_benchmark.benchmark_tools.dbms.parse_plan import parse_plan
from utils.utils import t_print


def check_valid(curr_statistics, dbms: DatabaseSystem, sql_query: str, min_runtime=100, verbose=True, ):
    # Timemouts are also a valid signal in learning
    if 'timeout' in curr_statistics and curr_statistics['timeout']:
        if verbose:
            t_print("Invalid since it ran into a timeout")
        return False

    try:
        analyze_plans = curr_statistics['analyze_plans']

        if analyze_plans is None or len(analyze_plans) == 0:
            if verbose:
                t_print("Unvalid because no analyze plans are available")
            return False

        analyze_plan = analyze_plans[0]

        if dbms == DatabaseSystem.POSTGRES:
            analyze_plan, ex_time, _ = parse_plan(analyze_plan, analyze=True, parse=True)
            analyze_plan.parse_lines_recursively()
            min_card = analyze_plan.min_card()

            min_ex_time = min([parse_plan(p, analyze=True, parse=True)[1] for p in analyze_plans])
        elif dbms == DatabaseSystem.DUCKDB:
            min_ex_time = min([float(p['result']) * 1000 for p in analyze_plans])  # convert to ms
            try:
                namespace_plan = json.loads(json.dumps(analyze_plan), object_hook=lambda d: SimpleNamespace(**d))
                parsed_plan = parse_plan_recursively(namespace_plan, include_zero_card=False,
                                                     table_id_mapping=None,
                                                     column_id_mapping=None, translate_table_col_names_to_id=False,
                                                     sql_query=sql_query)
                min_card = 1
            except ZeroCardFoundError as e:

                min_card = 0
        else:
            raise Exception(f"Unknown dbms {dbms}")

        if min_card == 0:
            if verbose:
                t_print("Unvalid because of zero cardinality")
            return False

        if min_ex_time < min_runtime:
            if verbose:
                t_print(f"Unvalid because of too short runtime ({min_ex_time} < {min_runtime})")
            return False

        return True
    except Exception as e:
        if verbose:
            traceback.print_exc()
            t_print(f"Unvalid due to error: {e} \n {curr_statistics}")
        return False
