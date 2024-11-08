from typing import Dict, List

from tqdm import tqdm

from card_est.run_wanderjoin import run_wj_card_est, get_wj_card_for_table_comb
from cross_db_benchmark.benchmark_tools.augment_deepdb_card import report_stats
from udf_graph.helper import extract_filter_join_conds_below_udf_from_plan
from udf_graph.utils import q_err


def annotate_wanderjoin_card(parsed_runs: Dict, schema_relationships, duckdb_kwargs: Dict = None,
                             pg_kwargs: Dict = None, multi_threaded: bool = True):
    q_stats = []
    for plan in tqdm(parsed_runs['parsed_plans']):
        filter_conds, join_conds, _ = extract_filter_join_conds_below_udf_from_plan(query_plan_node=plan,
                                                                                    udf_found=True,
                                                                                    assume_lazy_dbms=False,
                                                                                    database_statistics=parsed_runs[
                                                                                        'database_stats'],
                                                                                    schema_relationships=schema_relationships)

        # run wanderjoin
        if len(join_conds) > 0:
            wj_data = run_wj_card_est(filter_conds=filter_conds, join_conds=join_conds, duckdb_kwargs=duckdb_kwargs,
                                      pg_kwargs=pg_kwargs)
        else:
            wj_data = None

        # annotate cardinalities
        annotate_card_from_wj_data_to_plan(plan=plan, wj_data=wj_data, q_stats=q_stats)

    # report q-error and latency statistics
    report_stats(q_stats)

    return parsed_runs


def annotate_card_from_wj_data_to_plan(plan: Dict, wj_data: Dict, q_stats: List[Dict[str, float]]):
    if len(plan['children']) == 0:
        child_tables = [plan['plan_parameters']['table_name']]
    else:

        child_tables = []
        # iterate over children
        for child in plan['children']:
            child_tables.extend(annotate_card_from_wj_data_to_plan(plan=child, wj_data=wj_data, q_stats=q_stats))

    # extract cards for statistics calculation
    act_card = plan['plan_parameters']['act_card']
    est_card = plan['plan_parameters']['est_card']

    if plan['plan_parameters']['op_name'] in ['Aggregate', 'Partial Aggregate', 'UNGROUPED_AGGREGATE', 'AGGREGATE']:
        wj_card = est_card
    elif wj_data is None:
        if 'dd_est_card' in plan['plan_parameters']:
            dd_card = plan['plan_parameters']['dd_est_card']
            wj_card = dd_card
        else:
            wj_card = est_card
    else:
        # search for card for this node
        wj_card = get_wj_card_for_table_comb(tables=child_tables, wj_data=wj_data)

        if wj_card == 1:
            wj_card = est_card

    plan['plan_parameters']['wj_est_card'] = wj_card

    # compute the q-error
    q_err_wj = q_err(wj_card, act_card)
    q_err_dbms = q_err(est_card, act_card)

    # compute the q-error and latency
    q_stats.append({
        'q_errors_dbms': q_err_dbms,
        'q_errors_wj': q_err_wj,
        'latencies_ms': 0,
        'msg': '',
        'reduced_filters': 0,
    })

    # compute the card prod of children
    for card in ['wj_est']:
        card_prod = 1
        for child in plan['children']:
            card_prod *= child['plan_parameters'][f'{card}_card']
        plan['plan_parameters'][f'{card}_children_card'] = card_prod

    return child_tables
