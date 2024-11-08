import collections
import json
import os
from enum import Enum
from typing import Dict

import numpy as np
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.column_types import Datatype
from cross_db_benchmark.benchmark_tools.utils import load_schema_json, load_column_statistics, load_string_statistics


class Operator(Enum):
    NEQ = '!='
    EQ = '='
    LEQ = '<='
    GEQ = '>='
    LIKE = 'LIKE'
    NOT_LIKE = 'NOT LIKE'
    IS_NOT_NULL = 'IS NOT NULL'
    IS_NULL = 'IS NULL'
    IN = 'IN'

    # BETWEEN = 'BETWEEN'

    def __str__(self):
        return self.value


def negate_op(op: Operator) -> Operator:
    if op == Operator.EQ:
        return Operator.NEQ
    elif op == Operator.NEQ:
        return Operator.EQ
    elif op == Operator.LEQ:
        return Operator.GEQ
    elif op == Operator.GEQ:
        return Operator.LEQ
    elif op == Operator.LIKE:
        return Operator.NOT_LIKE
    elif op == Operator.NOT_LIKE:
        return Operator.LIKE
    elif op == Operator.IS_NULL:
        return Operator.IS_NOT_NULL
    elif op == Operator.IS_NOT_NULL:
        return Operator.IS_NULL
    else:
        raise ValueError(f'Cannot negate operator {op}')


class Aggregator(Enum):
    AVG = 'AVG'
    SUM = 'SUM'
    COUNT = 'COUNT'

    def __str__(self):
        return self.value


class ExtendedAggregator(Enum):
    MIN = 'MIN'
    MAX = 'MAX'

    def __str__(self):
        return self.value


def sample_acyclic_aggregation_query(column_stats, string_stats, group_by_threshold, int_neq_predicate_threshold,
                                     max_cols_per_agg, max_no_aggregates, max_no_group_by, max_no_joins,
                                     max_no_predicates, relationships_table, schema, randstate, complex_predicates,
                                     max_no_joins_static, max_no_aggregates_static, max_no_predicates_static,
                                     max_no_group_by_static, left_outer_join_ratio, groupby_limit_prob,
                                     groupby_having_prob, udf_counter):
    no_joins = randstate.randint(1, max_no_joins + 1)  # enforce joins by setting the lower bound to 1

    # make sure we have at least one predicate, aggregate or group by
    while True:
        no_predicates = randstate.randint(0, max_no_predicates + 1)
        no_aggregates = randstate.randint(0, max_no_aggregates + 1)
        no_group_bys = randstate.randint(0, max_no_group_by + 1)

        if no_predicates + no_aggregates + no_group_bys > 0:
            break

    # decide where to put the UDF
    lst = [no_predicates, no_aggregates, no_group_bys]

    while True:
        index = randstate.randint(0, len(lst))
        if lst[index] > 0:
            break

    udf_pred, udf_agg, udf_group = False, False, False
    if index == 0:
        udf_pred = True
    elif index == 1:
        udf_agg = True
    else:
        udf_group = True

    if max_no_joins_static:
        no_joins = max_no_joins
    if max_no_predicates_static:
        no_predicates = max_no_predicates
    if max_no_aggregates_static:
        no_aggregates = max_no_aggregates
    if max_no_group_by_static:
        no_group_bys = max_no_group_by

    start_t, joins, join_tables = sample_acyclic_join(no_joins, relationships_table, schema, randstate,
                                                      left_outer_join_ratio)

    numerical_aggregation_columns, possible_group_by_columns, predicates, possible_columns, udf_stats_pred = generate_predicates(
        column_stats,
        complex_predicates,
        group_by_threshold,
        int_neq_predicate_threshold,
        join_tables,
        no_predicates, randstate,
        string_stats, udf_counter, udf_pred)
    limit = None
    if randstate.rand() < groupby_limit_prob:
        limit = randstate.choice([10, 100, 1000])

    group_bys, udf_stats_gb = sample_group_bys(no_group_bys, possible_group_by_columns, randstate, udf_counter,
                                               udf_group, possible_columns)
    aggregations, udf_stats_agg = sample_aggregations(max_cols_per_agg, no_aggregates, numerical_aggregation_columns,
                                                      randstate,
                                                      complex_predicates=complex_predicates, udf_counter=udf_counter,
                                                      use_udf=udf_agg, possible_columns=possible_columns)

    # Make sure that we only have a UDF at either a predicate, either a group by, either an aggregation
    assert sum([bool(udf_stats_agg), bool(udf_stats_gb), bool(udf_stats_pred)]) == 1
    assert len(set(udf_stats_pred.keys()).intersection(
        set(udf_stats_agg.keys()))) == 0, f'{udf_stats_pred.keys()}\n{udf_stats_agg.keys()}'
    assert len(set(udf_stats_pred.keys()).intersection(
        set(udf_stats_gb.keys()))) == 0, f'{udf_stats_pred.keys()}\n{udf_stats_gb.keys()}'

    # combine the dictionaries from UDF stats for the different types
    udf_stats = {**udf_stats_pred, **udf_stats_agg, **udf_stats_gb}
    assert len(udf_stats) == len(udf_stats_pred) + len(udf_stats_agg) + len(
        udf_stats_gb), f'{udf_stats}\n{udf_stats_pred}\n{udf_stats_agg}\n{udf_stats_gb}'  # make sure that we had no duplicates in the UDF stats

    having_clause = None
    if randstate.rand() < groupby_having_prob:
        idx = randstate.randint(0, len(aggregations))
        _, cols = aggregations[idx]
        literal = sum([vars(vars(column_stats)[col[0]])[col[1]].mean for col in cols])
        op = rand_choice(randstate, [Operator.LEQ, Operator.GEQ, Operator.NEQ])
        having_clause = (idx, literal, op)

    q = GenQuery(aggregations, group_bys, joins, predicates, start_t, list(join_tables), limit=limit,
                 having_clause=having_clause)

    return q, udf_stats


def generate_predicates(column_stats, complex_predicates, group_by_threshold, int_neq_predicate_threshold, join_tables,
                        no_predicates, randstate, string_stats, udf_counter, use_udf):
    numerical_aggregation_columns, possible_columns, possible_string_columns, \
        possible_group_by_columns, table_predicates, string_table_predicates = analyze_columns(
        column_stats, group_by_threshold, join_tables, string_stats, complex_predicates)
    if complex_predicates:
        raise Exception("Complex predicates not supported w. UDFs")
        # predicates = sample_complex_predicates(column_stats, string_stats, int_neq_predicate_threshold, no_predicates,
        #                                        possible_columns,
        #                                        possible_string_columns, table_predicates, string_table_predicates,
        #                                        randstate)
    else:
        predicates, udf_stats = sample_predicates(column_stats, int_neq_predicate_threshold, no_predicates,
                                                  possible_columns,
                                                  table_predicates, randstate, udf_name_count=udf_counter,
                                                  use_udf=use_udf)
        return numerical_aggregation_columns, possible_group_by_columns, predicates, possible_columns, udf_stats


class GenQuery:

    def __init__(self, aggregations, group_bys, joins, predicates, start_t, join_tables, alias_dict=None,
                 inner_groupby=None, subquery_alias=None, limit=None, having_clause=None):
        if alias_dict is None:
            alias_dict = dict()
        self.aggregations = aggregations
        self.group_bys = group_bys
        self.joins = joins
        self.predicates = predicates
        self.start_t = start_t
        self.join_tables = join_tables
        self.alias_dict = alias_dict
        self.exists_predicates = []
        self.inner_groupby = inner_groupby
        self.subquery_alias = subquery_alias
        self.limit = limit
        self.having_clause = having_clause
        if self.inner_groupby is not None:
            self.alias_dict = {t: subquery_alias for t in self.join_tables}

    def append_exists_predicate(self, q_rec, not_exist):
        self.exists_predicates.append((q_rec, not_exist))

    def generate_sql_query(self, udf_stats: Dict, semicolon=True):
        # group_bys
        group_by_str = ''
        order_by_str = ''

        group_by_cols = []
        if len(self.group_bys) > 0:
            # group_by_cols = [f'"{table}"."{column}"' for table, column, _ in self.group_bys]
            for gb_col in self.group_bys:
                if len(gb_col) > 3:  # in this case we have also information about the UDF columns
                    udf_param_str = ""
                    for udf_incol in gb_col[4]:
                        udf_param_str += f'"{gb_col[0]}"."{udf_incol}",'
                    group_by_cols.append(f'{gb_col[3]}({udf_param_str[:-1]})')
                else:
                    group_by_cols.append(f'"{gb_col[0]}"."{gb_col[1]}"')

            group_by_col_str = ', '.join(group_by_cols)
            group_by_str = f' GROUP BY {group_by_col_str}'
            order_by_str = f' ORDER BY {group_by_col_str}'

        # aggregations
        # Hacky solution but it serves the purpose of this prototype
        # We know that we have 1 or 0 aggregations
        # If we have a UDF "inside" the aggregation, we know that this tuple has > 2 elements
        if len(self.aggregations) > 0 and len(self.aggregations[0]) > 2:
            assert len(self.aggregations) <= 1, f'{self.aggregations}'
            table = self.aggregations[0]['table']
            udf_input_col_strings = [f'"{table}"."{udf_incol}"' for udf_incol in self.aggregations[0]['in_cols']]
            udf_args = ','.join(udf_input_col_strings)

            assert len(udf_input_col_strings) == len(udf_stats[self.aggregations[0]["udf_name"]][
                                                         "incols"]), f'{udf_input_col_strings}\n{udf_stats[self.aggregations[0]["udf_name"]]["incols"]}'

            aggregation_str_list = [
                f'{self.aggregations[0]["agg"]}({self.aggregations[0]["udf_name"]}({udf_args}))']

        else:
            aggregation_str_list = []
            for i, (aggregator, columns) in enumerate(self.aggregations):
                if aggregator == Aggregator.COUNT:
                    aggregation_str_list.append(f'COUNT(*)')
                else:
                    agg_cols = ' + '.join([f'"{table}"."{col}"' for table, col in columns])
                    aggregation_str_list.append(f'{str(aggregator)}({agg_cols})')
        aggregation_str = ', '.join(group_by_cols + [f'{agg} as agg_{i}' for i, agg in enumerate(aggregation_str_list)])
        if aggregation_str == '':
            aggregation_str = '*'

        # having clause
        having_str = ""
        if self.having_clause is not None:
            idx, literal, op = self.having_clause
            having_str = f" HAVING {aggregation_str_list[idx]} {str(op)} {literal}"

        # predicates
        predicate_str = str(self.predicates)

        # other parts can simply be replaced with aliases
        for t, alias_t in self.alias_dict.items():
            predicate_str = predicate_str.replace(f'"{t}"', alias_t)
            aggregation_str = aggregation_str.replace(f'"{t}"', alias_t)
            group_by_str = group_by_str.replace(f'"{t}"', alias_t)
            order_by_str = order_by_str.replace(f'"{t}"', alias_t)
            having_str = having_str.replace(f'"{t}"', alias_t)

        if len(self.exists_predicates) > 0:
            exists_preds = []
            for q_rec, not_exist in self.exists_predicates:
                if not_exist:
                    exists_preds.append(f'NOT EXISTS ({q_rec.generate_sql_query(semicolon=False)})')
                else:
                    exists_preds.append(f'EXISTS ({q_rec.generate_sql_query(semicolon=False)})')
            exists_preds = ' AND '.join(exists_preds)
            if predicate_str == '':
                predicate_str += f' WHERE {exists_preds} '
            else:
                predicate_str += f' AND {exists_preds}'

        # join
        if self.inner_groupby is not None:
            join_str = f'({self.inner_groupby.generate_sql_query(semicolon=False)}) {self.subquery_alias}'

        else:
            already_repl = set()

            def repl_alias(t, no_alias_intro=False):
                if t in self.alias_dict:
                    alias_t = self.alias_dict[t]
                    if t in already_repl or no_alias_intro:
                        return alias_t

                    else:
                        return f'"{t}" {alias_t}'

                return f'"{t}"'

            # Modify this code so that we do not have AND in the join condition as this causes lots of
            # headache with the parser when we also have UDFs involved
            join_str = repl_alias(self.start_t)
            for table_l, column_l, table_r, column_r, left_outer in self.joins:
                join_kw = "JOIN" if not left_outer else "LEFT OUTER JOIN"
                join_str += f' {join_kw} {repl_alias(table_r)}'
                join_cond = f'{repl_alias(table_l, no_alias_intro=True)}."{column_l[0]}" = 'f'{repl_alias(table_r, no_alias_intro=True)}."{column_r[0]}"'
                # join_cond = ' AND '.join([f'{repl_alias(table_l, no_alias_intro=True)}."{col_l}" = '
                #                           f'{repl_alias(table_r, no_alias_intro=True)}."{col_r}"'
                #                           for col_l, col_r in zip(column_l, column_r)])
                join_str += f' ON {join_cond}'

        limit_str = ""
        if self.limit is not None:
            limit_str = f" LIMIT {self.limit}"

        # don't use order by
        order_by_str = ""

        sql_query = f"SELECT {aggregation_str} FROM {join_str} {predicate_str}{group_by_str}{having_str}{order_by_str}{limit_str}".strip()

        if semicolon:
            sql_query += ';'

        return sql_query


def generate_workload(dataset, target_path, udf_stats_path=None, col_stats_dir: str = None, num_queries=100,
                      max_no_joins=3, max_no_predicates=3,
                      max_no_aggregates=3,
                      max_no_group_by=3, max_cols_per_agg=2, group_by_threshold=10000, int_neq_predicate_threshold=100,
                      seed=0, complex_predicates=False, force=False, max_no_joins_static=False,
                      max_no_aggregates_static=False, max_no_predicates_static=False, max_no_group_by_static=False,
                      left_outer_join_ratio=0.0, groupby_limit_prob=0.0, groupby_having_prob=0.0,
                      exists_predicate_prob=0.0, max_no_exists=0, outer_groupby_prob=0.0):
    randstate = np.random.RandomState(seed)
    udf_dict = {}

    if os.path.exists(target_path) and not force:
        print("Workload already generated")
        return

    # read the schema file
    column_stats = load_column_statistics(col_stats_dir=col_stats_dir, dataset=dataset)
    string_stats = load_string_statistics(col_stats_dir=col_stats_dir, dataset=dataset)
    schema = load_schema_json(dataset)

    # build index of join relationships
    relationships_table = collections.defaultdict(list)
    for table_l, column_l, table_r, column_r in schema.relationships:
        if not isinstance(column_l, list):
            column_l = [column_l]
        if not isinstance(column_r, list):
            column_r = [column_r]

        relationships_table[table_l].append([column_l, table_r, column_r])
        relationships_table[table_r].append([column_r, table_l, column_l])

    queries = []
    for i in tqdm(range(num_queries)):
        # sample query as long as it does not meet requirements
        tries = 0
        desired_query = False
        while not desired_query:
            q, udf_stats = sample_acyclic_aggregation_query(column_stats, string_stats, group_by_threshold,
                                                            int_neq_predicate_threshold,
                                                            max_cols_per_agg, max_no_aggregates, max_no_group_by,
                                                            max_no_joins,
                                                            max_no_predicates, relationships_table, schema, randstate,
                                                            complex_predicates, max_no_joins_static,
                                                            max_no_aggregates_static,
                                                            max_no_predicates_static, max_no_group_by_static,
                                                            left_outer_join_ratio, groupby_limit_prob,
                                                            groupby_having_prob, udf_counter=i)

            # retry maybe
            desired_query |= check_matches_criteria(q, complex_predicates, max_no_aggregates, max_no_aggregates_static,
                                                    max_no_group_by, max_no_group_by_static, max_no_joins,
                                                    max_no_joins_static, max_no_predicates, max_no_predicates_static,
                                                    udf_stats=udf_stats, col_stats=column_stats)

            # samples subqueries (self joins) for exists / not exists predicates and adds to query
            sample_exists_subqueries(column_stats, complex_predicates, exists_predicate_prob, group_by_threshold,
                                     int_neq_predicate_threshold, max_no_exists, q, randstate, relationships_table,
                                     string_stats)

            # potentially sample outer query with another group by
            outer_groupby = randstate.rand() < outer_groupby_prob
            if outer_groupby:
                q = sample_outer_groupby(complex_predicates, q, randstate)

            if desired_query:
                sql_query = q.generate_sql_query(udf_stats=udf_stats)
                queries.append(sql_query)
                udf_dict_len_before = len(udf_dict)
                udf_dict = {**udf_dict, **udf_stats}
                assert len(udf_dict) == udf_dict_len_before + len(udf_stats), f'{udf_dict.keys()}\n{udf_stats.keys()}'
                break
            else:
                tries += 1
                if tries > 10000:
                    raise ValueError("Did not find a valid query after 10000 trials. "
                                     "Please check if your conditions can be fulfilled")

    print(f'Writing {len(queries)} queries to {target_path}')
    target_dir = os.path.dirname(target_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    with open(target_path, "w") as text_file:
        text_file.write('\n'.join(queries))

    if udf_stats_path is not None:
        if not os.path.exists(os.path.dirname(udf_stats_path)):
            os.makedirs(os.path.dirname(udf_stats_path), exist_ok=True)
        with open(udf_stats_path, "w") as text_file:
            text_file.write(json.dumps(udf_dict))


def sample_outer_groupby(complex_predicates, q, randstate):
    subquery_alias = 'subgb'
    outer_aggs = []
    for i, (_, cols) in enumerate(q.aggregations):
        l = list(Aggregator)
        if complex_predicates:
            l += list(ExtendedAggregator)
        agg_type = rand_choice(randstate, l)
        outer_aggs.append((agg_type, [[subquery_alias, f'agg_{i}']]))
    outer_groupby = []
    if len(q.group_bys) > 0:
        outer_groupby = rand_choice(randstate, q.group_bys,
                                    no_elements=randstate.randint(0, len(q.group_bys)),
                                    replace=False)
        outer_groupby = [(subquery_alias, c, x) for _, c, x in outer_groupby]
    q = GenQuery(outer_aggs, outer_groupby, [], PredicateOperator(LogicalOperator.AND, []),
                 None, q.join_tables, inner_groupby=q, subquery_alias=subquery_alias)
    return q


def sample_exists_subqueries(column_stats, complex_predicates, exists_ratio, group_by_threshold,
                             int_neq_predicate_threshold, max_no_exists, q, randstate, relationships_table,
                             string_stats):
    exists_subquery = randstate.rand() < exists_ratio
    eligible_exist = list(set(q.join_tables).intersection(relationships_table.keys()))
    if exists_subquery and len(eligible_exist) > 0:

        no_exists = randstate.randint(1, max_no_exists + 1)

        alias_dict = dict()
        exist_tables = []
        chosen_aliases = set()

        for _ in range(no_exists):
            alias_table = randstate.choice(eligible_exist)

            if alias_table not in alias_dict:
                alias_dict[alias_table] = f'{alias_table.lower()}_1'
            chosen_aliases.add(alias_dict[alias_table])

            for i in range(2, int(1e10)):
                subquery_alias = f'{alias_table.lower()}_{i}'
                if subquery_alias not in chosen_aliases:
                    rec_alias_dict = {alias_table: subquery_alias}
                    exist_tables.append((alias_table, rec_alias_dict))
                    chosen_aliases.add(subquery_alias)
                    break

        q.alias_dict = alias_dict

        # for each table generate exists subquery
        for t, rec_alias_dict in exist_tables:
            no_rec_pred = randstate.randint(1, 3)
            _, _, predicates = generate_predicates(column_stats, complex_predicates, group_by_threshold,
                                                   int_neq_predicate_threshold, [t], no_rec_pred, randstate,
                                                   string_stats)
            possible_cols = set()
            for ct, _, _ in relationships_table[t]:
                possible_cols.update(ct)
            if len(possible_cols) == 0:
                continue
            key_exist_col = randstate.choice(list(possible_cols))

            op = randstate.choice([Operator.EQ, Operator.NEQ])
            self_pred = ColumnPredicate(t, key_exist_col, op, f'{alias_dict[t]}."{key_exist_col}"')
            if type(predicates) == ColumnPredicate or len(predicates.children) > 0:
                p = PredicateOperator(LogicalOperator.AND, [predicates, self_pred])
            else:
                p = self_pred

            q_rec = GenQuery([], [], [], p, t, [t], alias_dict=rec_alias_dict)
            q.append_exists_predicate(q_rec, randstate.choice([True, False]))


def check_matches_criteria(q, complex_predicates, max_no_aggregates,
                           max_no_aggregates_static, max_no_group_by, max_no_group_by_static, max_no_joins,
                           max_no_joins_static, max_no_predicates, max_no_predicates_static, udf_stats, col_stats):
    desired_query = True
    if (max_no_joins_static and len(q.joins) < max_no_joins) or \
            (max_no_aggregates_static and len(q.aggregations) < max_no_aggregates) or \
            (max_no_group_by_static and len(q.group_bys) < max_no_group_by):
        desired_query = False
    if max_no_predicates_static:
        if complex_predicates:
            raise NotImplementedError("Check not implemented for complex predicates")
        else:
            if len(q.predicates.children) != max_no_predicates:
                desired_query = False

    if not "func_" in q.generate_sql_query(udf_stats=udf_stats):
        desired_query = False

    # make sure we have only one udf per query
    assert len(udf_stats.keys()) == 1
    # we know that the dict has only one element since we only have one udf per query
    func_name = None
    for key in udf_stats:
        func_name = key
        break
    assert func_name is not None

    # make sure that columns are only used that have a low nan ratio since we filter for that value when generating the udf
    for col_name, col_stats in vars(vars(col_stats)[udf_stats[func_name]["table"]]).items():
        if col_name in udf_stats[func_name]["incols"] and (col_stats.nan_ratio > 0.2):
            desired_query = False
            break

    if udf_stats[func_name]["outcol"] == []:
        desired_query = False

    # Ensure that the UDF is applied to the start table to avoid possible
    # problems, e.g., with LEFT OUTER JOINs
    func_name = ""
    for k in udf_stats.keys():
        func_name = k

    if udf_stats[func_name]["table"] != q.start_t:
        desired_query = False

    return desired_query


def sample_group_bys(no_group_bys, possible_group_by_columns, randstate, udf_counter, use_udf, possible_columns):
    group_bys = []
    udf_dict = {}
    if no_group_bys > 0:

        if use_udf:
            udf_name = "func_" + str(udf_counter)
            no_group_bys = min(no_group_bys, len(possible_group_by_columns))
            group_bys = rand_choice(randstate, possible_group_by_columns, no_elements=no_group_bys, replace=False)
            incols = [group_bys[0][1]] + get_random_udf_params(possible_columns, randstate, group_bys[0][0])
            incols = list(set(incols))  # make sure we have no duplicates
            group_bys[0].append(udf_name)  # we can do that since we only have one group by
            group_bys[0].append(incols)  # we can do that since we only have one group by
            udf_dict = {udf_name: {"incols": incols, "outcol": group_bys[0][1], "table": group_bys[0][0]}}
        else:
            no_group_bys = min(no_group_bys, len(possible_group_by_columns))
            group_bys = rand_choice(randstate, possible_group_by_columns, no_elements=no_group_bys, replace=False)
            # group_bys = randstate.sample(possible_group_by_columns, no_group_bys)
        return group_bys, udf_dict
    else:
        return group_bys, udf_dict


def sample_aggregations(max_cols_per_agg, no_aggregates, numerical_aggregation_columns, randstate,
                        complex_predicates=False, use_udf=False, udf_counter=0, possible_columns=[]):
    aggregations = []
    udf_dict = {}
    if no_aggregates > 0:
        if use_udf:
            udf_name = "func_" + str(udf_counter)
            assert no_aggregates == 1, f'Only one aggregate supported for UDFs'
            for i in range(no_aggregates):
                # if there are no numerical columns, we can only use count
                if len(numerical_aggregation_columns) == 0:
                    table = rand_choice(randstate, possible_columns, no_elements=1, replace=False)[0]
                    incols = list(set(get_random_udf_params(possible_columns, randstate, table)))
                    udf_out_col = rand_choice(randstate, incols, no_elements=1, replace=False)
                    udf_dict = {udf_name: {"incols": incols, "outcol": udf_out_col, "table": table}}
                    aggregations.append(
                        {'agg': Aggregator.COUNT, 'udf_name': udf_name, 'in_cols': incols, 'table': table})
                else:
                    udf_out_col = rand_choice(randstate, numerical_aggregation_columns, no_elements=1, replace=False)
                    incols = list(set([udf_out_col[0][1]] + get_random_udf_params(possible_columns, randstate,
                                                                                  udf_out_col[0][0])))
                    udf_dict = {udf_name: {"incols": incols, "outcol": udf_out_col[0][1], "table": udf_out_col[0][0]}}
                    l = list(Aggregator)
                    agg = rand_choice(randstate, l)
                    aggregations.append(
                        {'agg': agg, 'udf_name': udf_name, 'in_cols': incols, 'table': udf_out_col[0][0]})
        else:
            for i in range(no_aggregates):
                no_agg_cols = min(randstate.randint(1, max_cols_per_agg + 1), len(numerical_aggregation_columns))
                l = list(Aggregator)
                if complex_predicates:
                    l += list(ExtendedAggregator)
                agg = rand_choice(randstate, l)
                cols = rand_choice(randstate, numerical_aggregation_columns, no_elements=no_agg_cols, replace=False)
                # cols = randstate.sample(numerical_aggregation_columns, no_agg_cols)
                if agg == Aggregator.COUNT:
                    cols = []
                if no_agg_cols == 0 and agg != Aggregator.COUNT:
                    continue

                aggregations.append((agg, cols))
            if len(aggregations) == 0:
                aggregations.append((Aggregator.COUNT, []))
    return aggregations, udf_dict


class ColumnPredicate:
    def __init__(self, table, col_name, operator, literal, udf_incols=[], udf_name=None):
        self.table = table
        self.col_name = col_name
        self.operator = operator
        self.literal = literal
        self.udf_incols = udf_incols
        self.udf_name = udf_name

    def __str__(self):
        return self.to_sql(top_operator=True)

    def to_sql(self, top_operator=False):
        if len(self.udf_incols) == 0:
            if self.operator == Operator.IS_NOT_NULL:
                predicates_str = f'"{self.table}"."{self.col_name}" IS NOT NULL'
            elif self.operator == Operator.IS_NULL:
                predicates_str = f'"{self.table}"."{self.col_name}" IS NULL'
            else:
                predicates_str = f'"{self.table}"."{self.col_name}" {str(self.operator)} {self.literal}'

            if top_operator:
                predicates_str = f' WHERE {predicates_str}'
        else:
            udf_call_str = ','.join([f'"{self.table}"."{col}"' for col in self.udf_incols])
            udf_call_str = f'{self.udf_name}({udf_call_str})'

            if self.operator == Operator.IS_NOT_NULL:
                predicates_str = f'{udf_call_str} IS NOT NULL'
            elif self.operator == Operator.IS_NULL:
                predicates_str = f'{udf_call_str} IS NULL'
            else:
                predicates_str = f'{udf_call_str} {str(self.operator)} {self.literal}'

            if top_operator:
                predicates_str = f' WHERE {predicates_str}'

        return predicates_str


class LogicalOperator(Enum):
    AND = 'AND'
    OR = 'OR'

    def __str__(self):
        return self.value


class PredicateOperator:
    def __init__(self, logical_op, children=None):
        self.logical_op = logical_op
        if children is None:
            children = []
        self.children = children

    def __str__(self):
        return self.to_sql(top_operator=True)

    def to_sql(self, top_operator=False):
        sql = ''
        if len(self.children) > 0:
            # if len(self.children) == 1:
            #     return self.children[0].to_sql(top_operator=top_operator)

            predicates_str_list = [c.to_sql() for c in self.children]
            sql = f' {str(self.logical_op)} '.join(predicates_str_list)

            if top_operator:
                sql = f' WHERE {sql}'
            elif len(self.children) > 1:
                sql = f'({sql})'

        return sql


class PredicateChain(Enum):
    SIMPLE = 'SIMPLE'
    OR_OR = 'OR_OR'
    OR = 'OR'
    OR_AND = 'OR_AND'

    def __str__(self):
        return self.value


def sample_complex_predicates(column_stats, string_stats, int_neq_predicate_threshold, no_predicates, possible_columns,
                              possible_string_columns, table_predicates, string_table_predicates, randstate,
                              p_or=0.05, p_or_or=0.05, p_or_and=0.05, p_second_column=0.5):
    # weight the prob of being sampled by number of columns in table
    # make sure we do not just have conditions on one table with many columns
    weights = [1 / table_predicates[t] for t, col_name in possible_columns]
    weights += [1 / string_table_predicates[t] for t, col_name in possible_string_columns]
    weights = np.array(weights)
    weights /= np.sum(weights)

    possible_columns += possible_string_columns
    no_predicates = min(no_predicates, len(possible_columns))
    predicate_col_idx = randstate.choice(range(len(possible_columns)), no_predicates, p=weights, replace=False)
    predicate_columns = [possible_columns[i] for i in predicate_col_idx]
    predicates = []
    for [t, col_name] in predicate_columns:

        # sample which predicate chain
        predicate_options = [PredicateChain.SIMPLE, PredicateChain.OR, PredicateChain.OR_OR, PredicateChain.OR_AND]
        pred_weights = [1 - p_or - p_or_or - p_or_and, p_or, p_or_or, p_or_and]
        pred_chain_idx = randstate.choice(range(len(predicate_options)), 1, p=pred_weights)[0]
        pred_chain = predicate_options[pred_chain_idx]

        # sample first predicate
        p = sample_predicate(string_stats, column_stats, t, col_name, int_neq_predicate_threshold, randstate,
                             complex_predicate=True)
        if p is None:
            continue

        if pred_chain == PredicateChain.SIMPLE:
            predicates.append(p)
        else:
            # sample if we use another column condition
            second_column = randstate.uniform() < p_second_column
            if second_column:
                potential_2nd_col = [c2 for t2, c2 in possible_columns
                                     if t2 == t and c2 != col_name and [t2, c2] not in predicate_columns]
                if len(potential_2nd_col) == 0:
                    continue
                second_col = rand_choice(randstate, potential_2nd_col)
                p2 = sample_predicate(string_stats, column_stats, t, second_col, int_neq_predicate_threshold, randstate,
                                      complex_predicate=True)
            else:
                p2 = sample_predicate(string_stats, column_stats, t, col_name, int_neq_predicate_threshold, randstate,
                                      complex_predicate=True)
            if p2 is None:
                continue

            complex_pred = None
            if pred_chain == PredicateChain.OR:
                complex_pred = PredicateOperator(LogicalOperator.OR, [p, p2])
            else:
                p3 = sample_predicate(string_stats, column_stats, t, col_name, int_neq_predicate_threshold, randstate,
                                      complex_predicate=True)
                if p3 is None:
                    complex_pred = PredicateOperator(LogicalOperator.OR, [p, p2])
                else:
                    if pred_chain == PredicateChain.OR_OR:
                        complex_pred = PredicateOperator(LogicalOperator.OR, [p, p2, p3])
                    elif pred_chain == PredicateChain.OR_AND:
                        complex_pred = PredicateOperator(LogicalOperator.OR, [p, PredicateOperator(LogicalOperator.AND,
                                                                                                   [p2, p3])])
            predicates.append(complex_pred)

    if len(predicates) == 1:
        return predicates[0]

    return PredicateOperator(LogicalOperator.AND, predicates)


def sample_predicate(string_stats, column_stats, t, col_name, int_neq_predicate_threshold, randstate,
                     complex_predicate=False, p_like=0.5, p_is_not_null=0.1, p_in=0.5, p_between=0.3,
                     p_not_like=0.5, p_mid_string_whitespace=0.5, udf_incols=[], udf_name=None):
    col_stats = vars(vars(column_stats)[t]).get(col_name)
    str_stats = None
    if string_stats is not None:
        str_stats = vars(vars(string_stats)[t]).get(col_name)

    if complex_predicate:

        # LIKE / NOT LIKE
        if col_stats is None or col_stats.datatype == str(Datatype.MISC) or \
                (str_stats is not None and randstate.uniform() < p_like):
            freq_words = [w for w in str_stats.freq_str_words if len(w) > 1]
            if len(freq_words) == 0:
                return None
            literal = rand_choice(randstate, freq_words)

            # additional whitespace in the middle
            if randstate.uniform() < p_mid_string_whitespace:
                split_pos = randstate.randint(1, len(literal))
                literal = literal[:split_pos] + '%' + literal[split_pos:]

            literal = f"'%{literal}%'"

            if randstate.uniform() < p_not_like:
                op = Operator.NOT_LIKE
            else:
                op = Operator.LIKE

            return ColumnPredicate(t, col_name, op, literal)

        # IS NOT NULL / IS NULL
        if col_stats.nan_ratio > 0 and randstate.uniform() < p_is_not_null:
            if randstate.uniform() < 0.8:
                return ColumnPredicate(t, col_name, Operator.IS_NOT_NULL, None)
            return ColumnPredicate(t, col_name, Operator.IS_NULL, None)

        # IN
        if col_stats.datatype == str(Datatype.CATEGORICAL) and randstate.uniform() < p_in:
            # rand_choice(randstate, l, no_elements=None, replace=False)
            literals = col_stats.unique_vals
            first_cap = min(len(literals), 10)
            literals = literals[:first_cap]

            if len(literals) <= 1:
                return None

            no_in_literals = randstate.randint(1, len(literals))
            literals = rand_choice(randstate, literals, no_elements=no_in_literals, replace=False)
            literals = ', '.join([f"'{l}'" for l in literals])
            literals = f'({literals})'

            return ColumnPredicate(t, col_name, Operator.IN, literals)

        if col_stats.datatype in {str(Datatype.INT), str(Datatype.FLOAT)} and randstate.uniform() < p_between:
            l1 = sample_literal_from_percentiles(col_stats.percentiles, randstate,
                                                 round=col_stats.datatype == str(Datatype.INT))
            l2 = sample_literal_from_percentiles(col_stats.percentiles, randstate,
                                                 round=col_stats.datatype == str(Datatype.INT))
            if l1 == l2:
                l2 += 1
            literal = f'{min(l1, l2)} AND {max(l1, l2)}'
            return ColumnPredicate(t, col_name, Operator.BETWEEN, literal)

    # simple predicates
    if col_stats.datatype == str(Datatype.INT):
        reasonable_ops = [Operator.LEQ, Operator.GEQ]
        if col_stats.num_unique < int_neq_predicate_threshold:
            reasonable_ops.append(Operator.EQ)
            reasonable_ops.append(Operator.NEQ)

        literal = sample_literal_from_percentiles(col_stats.percentiles, randstate, round=True)

    elif col_stats.datatype == str(Datatype.FLOAT):
        reasonable_ops = [Operator.LEQ, Operator.GEQ]
        literal = sample_literal_from_percentiles(col_stats.percentiles, randstate, round=False)
        # nan comparisons only produce errors
        # happens when column is all nan
        if np.isnan(literal):
            return None
    elif col_stats.datatype == str(Datatype.CATEGORICAL):
        reasonable_ops = [Operator.EQ, Operator.NEQ]
        possible_literals = [v for v in col_stats.unique_vals if v is not None and
                             not (isinstance(v, float) and np.isnan(v))]
        if len(possible_literals) == 0:
            return None
        literal = rand_choice(randstate, possible_literals)
        literal = f"'{literal}'"
    else:
        raise NotImplementedError
    operator = rand_choice(randstate, reasonable_ops)
    return ColumnPredicate(t, col_name, operator, literal, udf_incols=udf_incols, udf_name=udf_name)


def get_random_udf_params(possible_cols, randstate, table):
    def filter_for_tab(possible_cols, table):
        col_set = []
        for col in possible_cols:
            if col[0] == table:
                col_set.append(col[1])
        return col_set

    col_candidates = filter_for_tab(possible_cols, table)
    rand_col_idxs = randstate.choice(range(len(col_candidates)), np.random.randint(0, len(col_candidates), 1),
                                     replace=False)
    udf_params = [col_candidates[i] for i in rand_col_idxs]
    return udf_params


def sample_predicates(column_stats, int_neq_predicate_threshold, no_predicates, possible_columns, table_predicates,
                      randstate, udf_name_count, use_udf):
    # sample random predicates
    # weight the prob of being sampled by number of columns in table
    # make sure we do not just have conditions on one table with many columns
    weights = np.array([1 / table_predicates[t] for t, col_name in possible_columns])
    weights /= np.sum(weights)
    # we cannot sample more predicates than available columns
    no_predicates = min(no_predicates, len(possible_columns))
    predicate_col_idx = randstate.choice(range(len(possible_columns)), no_predicates, p=weights, replace=False)
    predicate_columns = [possible_columns[i] for i in predicate_col_idx]
    predicates = []
    udf_predicate = use_udf
    udf_dict = {}
    for [t, col_name] in predicate_columns:
        if udf_predicate:  # only one predicate containing a UDF is used
            udf_name = "func_" + str(udf_name_count)
            incols = [col_name] + get_random_udf_params(possible_columns, randstate, t)
            incols = list(set(incols))  # make sure we have no duplicates
            p = sample_predicate(None, column_stats, t, col_name, int_neq_predicate_threshold, randstate,
                                 complex_predicate=False, udf_incols=incols, udf_name=udf_name)
            udf_dict = {udf_name: {"incols": incols, "outcol": col_name, "table": t}}
            udf_predicate = False
        else:
            p = sample_predicate(None, column_stats, t, col_name, int_neq_predicate_threshold, randstate,
                                 complex_predicate=False)
        if p is not None:
            predicates.append(p)

    return PredicateOperator(LogicalOperator.AND, predicates), udf_dict


def analyze_columns(column_stats, group_by_treshold, join_tables, string_stats, complex_predicates):
    # find possible columns for predicates
    possible_columns = []
    possible_string_columns = []
    possible_group_by_columns = []
    numerical_aggregation_columns = []
    # also track how many columns we have per table to reweight them
    table_predicates = collections.defaultdict(int)
    string_table_predicates = collections.defaultdict(int)
    for t in join_tables:
        for col_name, col_stats in vars(vars(column_stats)[t]).items():
            if col_stats.datatype in {str(d) for d in [Datatype.INT, Datatype.FLOAT, Datatype.CATEGORICAL]}:
                possible_columns.append([t, col_name])
                table_predicates[t] += 1

            if complex_predicates and col_name in vars(vars(string_stats)[t]):
                possible_string_columns.append([t, col_name])
                string_table_predicates[t] += 1

            # group by columns
            if col_stats.datatype in {str(d) for d in [Datatype.INT, Datatype.CATEGORICAL]} \
                    and col_stats.num_unique < group_by_treshold:
                possible_group_by_columns.append([t, col_name, col_stats.num_unique])

            # numerical aggregation columns
            if col_stats.datatype in {str(d) for d in [Datatype.INT, Datatype.FLOAT]}:
                numerical_aggregation_columns.append([t, col_name])
    return numerical_aggregation_columns, possible_columns, possible_string_columns, \
        possible_group_by_columns, table_predicates, string_table_predicates


def sample_literal_from_percentiles(percentiles, randstate, round=False):
    start_idx = randstate.randint(0, len(percentiles) - 1)
    if np.all(np.isnan(percentiles)):
        return np.nan
    literal = randstate.uniform(percentiles[start_idx], percentiles[start_idx + 1])
    if round:
        literal = int(literal)
    return literal


def rand_choice(randstate, l, no_elements=None, replace=False):
    if no_elements is None:
        idx = randstate.randint(0, len(l))
        return l[idx]
    else:
        idxs = randstate.choice(range(len(l)), no_elements, replace=replace)
        return [l[i] for i in idxs]


def sample_acyclic_join(no_joins, relationships_table, schema, randstate, left_outer_join_ratio):
    # randomly sample join
    joins = list()
    start_t = rand_choice(randstate, schema.tables)
    join_tables = {start_t}

    for i in range(no_joins):
        possible_joins = find_possible_joins(join_tables, relationships_table)

        # randomly select one join
        if len(possible_joins) > 0:
            t, column_l, table_r, column_r = rand_choice(randstate, possible_joins)
            join_tables.add(table_r)

            left_outer_join = False
            if left_outer_join_ratio > 0 and randstate.rand() < left_outer_join_ratio:
                left_outer_join = True

            joins.append((t, column_l, table_r, column_r, left_outer_join))
        else:
            break
    return start_t, joins, join_tables


def find_possible_joins(join_tables, relationships_table):
    possible_joins = list()
    for t in join_tables:
        for column_l, table_r, column_r in relationships_table[t]:
            if table_r in join_tables:
                continue
            possible_joins.append((t, column_l, table_r, column_r))
    return possible_joins
