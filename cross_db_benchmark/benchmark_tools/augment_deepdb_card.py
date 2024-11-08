import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, List, Tuple, Set

import numpy as np
from deepdb.ensemble_compilation.graph_representation import UnexpectedJoinCondError
from deepdb.inference import DeepDBEstimator, FilterCond
from tqdm import tqdm

from udf_graph.helper import extract_filter_conditions_recursively, parse_join_condition, join_cond_to_sql
from udf_graph.utils import q_err, extract_where_conditions


def annotate_deepdb_card(parsed_runs: Dict, deepdb_estimator: Optional[DeepDBEstimator], schema_relationships,
                         strip_plans_w_unexp_join_cond: bool = True):
    if deepdb_estimator is None:
        print("No deepdb instances provided. Skipping annotation.")
        return parsed_runs

    print("Annotating deepdb cardinalities...")
    q_stats = []
    plans_to_remove = []
    for plan in tqdm(parsed_runs['parsed_plans']):
        sql_conds = extract_where_conditions(plan['query'], target_table=None)
        try:
            augment_bottom_up_recursively(plan=plan, deepdb_instance=deepdb_estimator,
                                          database_statistics=parsed_runs['database_stats'], q_stats=q_stats,
                                          schema_relationships=schema_relationships, sql_conds=sql_conds,
                                          throw_unexp_jon_exc=strip_plans_w_unexp_join_cond)
        except UnexpectedJoinCondError as join_error:
            assert strip_plans_w_unexp_join_cond
            plans_to_remove.append(plan)

    if strip_plans_w_unexp_join_cond:
        for plan in plans_to_remove:
            parsed_runs['parsed_plans'].remove(plan)
        print(f"Removed {len(plans_to_remove)} plans with unexpected join conditions.")

    # report q-error and latency statistics
    report_stats(q_stats)

    return parsed_runs


def filter_out_redundant_conds(conditions: List[FilterCond], sql_conditions: List[FilterCond], schema_relationships):
    # dictionary mapping table,col to conditions
    cond_dict = defaultdict(list)
    for cond in conditions:
        cond_dict[(cond.table_name, cond.column_name)].append(cond)

    # remove redundant is-not-null conditions (when already filtering for a value)
    for col, conds in cond_dict.items():
        if len(conds) > 1:
            cleaned_conds = conds[:]
            for cond in conds:
                if cond.operator == 'IS NOT NULL':
                    cleaned_conds.remove(cond)

            cond_dict[col] = cleaned_conds

    # check for redundant conditions
    for t1, c1s, t2, c2s in schema_relationships:
        for c1 in c1s:
            for c2 in c2s:
                if (t1, c1) in cond_dict and (t2, c2) in cond_dict:
                    # go over conditions of both columns and remove conditions with same literals
                    for cond1 in cond_dict[(t1, c1)]:
                        for cond2 in cond_dict[(t2, c2)]:
                            if cond1.operator == cond2.operator and cond1.value == cond2.value:
                                cond_dict[(t2, c2)].remove(cond2)
                                break

    filtered_conditions = []
    for key, conds in cond_dict.items():
        filtered_conditions.extend(conds)

    return filtered_conditions


def augment_bottom_up_recursively(plan: Dict, deepdb_instance: DeepDBEstimator, database_statistics: Dict,
                                  q_stats: List[Dict[str, float]], schema_relationships: List,
                                  sql_conds: List[FilterCond], throw_unexp_jon_exc: bool) -> Tuple[
    List[FilterCond], Set[str], Dict[str, float], List[str]]:
    # go over children
    children_filters = []
    children_join_conds = []
    children_tables = []
    children_cards = []
    for child in plan['children']:
        # recursively call the function
        filters, tables, cards, join_conds = augment_bottom_up_recursively(child, deepdb_instance, database_statistics,
                                                                           q_stats=q_stats,
                                                                           schema_relationships=schema_relationships,
                                                                           sql_conds=sql_conds,
                                                                           throw_unexp_jon_exc=throw_unexp_jon_exc)
        children_filters.extend(filters)
        children_join_conds.extend(join_conds)
        children_tables.append(tables)
        children_cards.append(cards)

    if 'join' in plan['plan_parameters']:
        assert len(children_tables) == 2, f"Join node should have exactly two children: {children_tables}"
        # search for join conditions in this node
        try:
            join = parse_join_condition(plan['plan_parameters']['join'], children_tables[0], children_tables[1],
                                        database_statistics, schema_relationships=schema_relationships)
        except Exception as e:
            print(plan)
            print(database_statistics)
            raise e

        children_join_conds.append(join)

    if len(children_tables) == 0:
        children_tables = set()
    else:
        children_tables = set.union(*children_tables)

    # extract filter conditions from the current node
    if 'filter_columns' in plan['plan_parameters'] and plan['plan_parameters']['filter_columns'] is not None:
        filter_columns = plan['plan_parameters']['filter_columns']
        filter_conditions = extract_filter_conditions_recursively(filter_columns, database_statistics)
        children_filters.extend(filter_conditions)

    if 'table_name' in plan['plan_parameters']:
        children_tables.add(plan['plan_parameters']['table_name'])

    # estimate cardinality with deepdb
    act_card = plan['plan_parameters']['act_card']
    est_card = plan['plan_parameters']['est_card']
    t0 = time.time()

    # check whether the node qualifies for deepdb estimation
    qualify = True
    msg = None
    # if len(children_filters)==0:
    #     qualify = False
    #     msg = "No filters"
    if plan['plan_parameters']['op_name'] in ['Aggregate', 'Partial Aggregate', 'UNGROUPED_AGGREGATE', 'AGGREGATE']:
        qualify = False
        msg = "Aggregate node"
    elif est_card <= 1000 and act_card <= 1000:
        qualify = False
        msg = "Cardinality too low"
    elif len(children_tables) == 1 and len(children_filters) == 0:
        qualify = True
        # msg = "Single table and no filters"

    if qualify:
        filtered_filters = filter_out_redundant_conds(children_filters, schema_relationships=schema_relationships,
                                                      sql_conditions=sql_conds)

        for f in filtered_filters:
            if f.column_name in ['class_', '2B', '3B', 'Month/Year']:
                raise ValueError(f"Unknown column name: {f.column_name} in filter conditions.\n{plan}")

        try:
            dd_est_card = deepdb_instance.estimate_card(tuple(filtered_filters), frozenset(children_tables),
                                                        join_conditions=tuple(
                                                            [join_cond_to_sql(j) for j in children_join_conds]))
            dd_est_card = deepdb_instance.scale * dd_est_card
        except UnexpectedJoinCondError as join_error:
            if throw_unexp_jon_exc:
                raise join_error
            print(join_error, flush=True)
            dd_est_card = est_card
            msg = 'DeepDB unexp join cond error'

        except Exception as est_card_error:
            print(f"Error in deepdb estimation: {est_card_error}", flush=True)
            traceback.print_exc()
            print(f"Filters: {filtered_filters}", flush=True)
            dd_est_card = est_card
            msg = "DeepDB error"

    else:
        filtered_filters = children_filters
        if len(children_tables) == 1 and len(children_filters) == 0:
            # use dataset statistics
            # extract corresponding table stats
            found = None

            if isinstance(database_statistics, dict):
                for entry in database_statistics['table_stats']:
                    if entry['table_name'] == list(children_tables)[0]:
                        found = entry
                        break
                assert found is not None
                dd_est_card = found['estimated_size']
            else:
                for entry in database_statistics.table_stats:
                    if entry.table_name == list(children_tables)[0]:
                        found = entry
                        break
                assert found is not None
                dd_est_card = found.estimated_size
        else:
            dd_est_card = est_card

    # compute the q-error
    q_err_deepdb = q_err(dd_est_card, act_card)
    q_err_dbms = q_err(est_card, act_card)

    # this was probably a bug, anyway rarely happens
    if q_err_deepdb > 100 * q_err_dbms:
        dd_est_card = est_card
        msg = "DeepDB error too high"

        q_err_deepdb = q_err(dd_est_card, act_card)

    plan['plan_parameters']['dd_est_card'] = dd_est_card
    t1 = time.time()

    # compute the q-error and latency
    q_stats.append({
        'q_errors_dbms': q_err_dbms,
        'q_errors_deepdb': q_err_deepdb,
        'latencies_ms': (t1 - t0) * 1000,
        'msg': msg,
        'reduced_filters': len(children_filters) - len(filtered_filters),
    })

    # compute the card prod of children
    for card in ['dd_est']:
        card_prod = 1
        for child_card in children_cards:
            card_prod *= child_card[f'{card}_card']
        plan['plan_parameters'][f'{card}_children_card'] = card_prod

    return children_filters, children_tables, {'dd_est_card': dd_est_card, 'est_card': est_card,
                                               'act_card': act_card}, children_join_conds


def report_stats(q_stats: List[Dict[str, float]]):
    if len(q_stats) > 0:
        def report_percentiles(key):
            if key not in q_stats[0]:
                return
            vals = np.array([q_s[key] for q_s in q_stats])
            print(f"{key}: p50={np.median(vals):.2f} p95={np.percentile(vals, 95):.2f} "
                  f"p99={np.percentile(vals, 99):.2f} pmax={np.max(vals):.2f}")

        report_percentiles('q_errors_dbms')
        report_percentiles('q_errors_wj')
        report_percentiles('q_errors_deepdb')
        report_percentiles('latencies_ms')
        report_percentiles('reduced_filters')

        # create a histogram of the msg entries
        msg_hist = defaultdict(int)
        for q_s in q_stats:
            msg_hist[q_s['msg']] += 1
        print(f"Message histogram (entries: {len(q_stats)}):")
        for msg, count in msg_hist.items():
            print(f"{msg}: {count}")
