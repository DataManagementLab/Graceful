import itertools
from typing import Dict
from typing import List

import networkx as nx
from deepdb.inference import FilterCond
from networkx.readwrite import json_graph

from card_est.wanderjoin import WanderJoin
from udf_graph.helper import JoinCond, join_cond_to_sql

SOURCE_NODE = tuple("s")

"""
Based on: https://github.com/parimarjan/sql_representation/blob/master/sql_rep/utils.py
and https://github.com/parimarjan/learned-cardinalities/blob/master/scripts/wanderjoin.py

"""


def get_f_str(f: FilterCond):
    if f.value is None:
        value_str = ''
    elif isinstance(f.value, int) or isinstance(f.value, float):
        value_str = f"{f.value}"
    elif f.operator in ['IS NOT NULL', 'IS NULL']:
        value_str = ''
    elif f.value.endswith('::date') or f.value.endswith('::DATE'):
        value_str = f.value[:-6]
        assert value_str.startswith('\'') and value_str.endswith('\'')
    elif f.value.endswith('::time') or f.value.endswith('::TIME'):
        value_str = f.value[:-6]
        assert value_str.startswith('\'') and value_str.endswith('\'')
    elif f.value.startswith('\'') and f.value.endswith('\''):
        value_str = f.value
    else:
        value_str = f'\'{f.value}\''

    return f"{f.table_name}.{f.column_name} {f.operator} {value_str}"


def create_join_graph(join_conds: List[JoinCond], filter_conds: List[FilterCond]):
    join_graph = nx.Graph()

    for j in join_conds:
        join_graph.add_edge(j.table_name1, j.table_name2, join_condition=j, join_condition_sql=join_cond_to_sql(j))

    # annotate the nodes with the filter conditions
    for t1 in join_graph.nodes():
        filter_matches = []
        for f in filter_conds:
            if f.table_name == t1:
                filter_matches.append(f)

        predicates_str = [get_f_str(f) for f in filter_matches]
        join_graph.nodes[t1]["predicates"] = predicates_str

    return join_graph


def connected_subgraphs(g):
    # for i in range(2, len(g)+1):
    for i in range(1, len(g) + 1):
        for nodes_in_sg in itertools.combinations(g.nodes, i):
            sg = g.subgraph(nodes_in_sg)
            if nx.is_connected(sg):
                yield tuple(sorted(sg.nodes))


def generate_subset_graph(g):
    subset_graph = nx.DiGraph()
    for csg in connected_subgraphs(g):
        subset_graph.add_node(csg, cardinality={'total': 2, 'expected': 1, 'actual': 1})

    # print(subset_graph.nodes)
    # pdb.set_trace()
    # group by size
    max_subgraph_size = max(len(x) for x in subset_graph.nodes)
    subgraph_groups = [[] for _ in range(max_subgraph_size)]
    for node in subset_graph.nodes:
        subgraph_groups[len(node) - 1].append(node)

    for g1, g2 in zip(subgraph_groups, subgraph_groups[1:]):
        for superset in g2:
            super_as_set = set(superset)
            for subset in g1:
                assert len(superset) == len(subset) + 1
                if set(subset) < super_as_set:
                    subset_graph.add_edge(superset, subset)

    return subset_graph


def run_wj_card_est(filter_conds: List[FilterCond], join_conds: List[JoinCond], duckdb_kwargs: Dict = None,
                    pg_kwargs: Dict = None, seed: int = 1234, ):
    qrep = get_qrep(filter_conds, join_conds)
    qrep["subset_graph"] = nx.OrderedDiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
    qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])
    wj = WanderJoin(dd_kwargs=duckdb_kwargs, pg_kwargs=pg_kwargs, verbose=False, walks_timeout=0.5, seed=
    seed, use_tries=True, trie_cache=None, max_succ_walks=100)

    if SOURCE_NODE in list(qrep["subset_graph"].nodes()):
        qrep["subset_graph"].remove_node(SOURCE_NODE)

    data = wj.get_counts(qrep)
    return data


def get_qrep(filter_conds: List[FilterCond], join_conds: List[JoinCond]) -> Dict:
    assert len(join_conds) > 0, "No join conditions found"
    join_graph = create_join_graph(filter_conds=filter_conds, join_conds=join_conds)
    subset_graph = generate_subset_graph(join_graph)

    # print("query has",
    #       len(join_graph.nodes), "relations,",
    #       len(join_graph.edges), "joins, and",
    #       len(subset_graph), " possible subsets.",
    #       "took:", time.time() - start)

    ret = {}
    ret["join_graph"] = nx.adjacency_data(join_graph)
    ret["subset_graph"] = nx.adjacency_data(subset_graph)
    return ret


def get_wj_card_for_table_comb(tables: List[str], wj_data: Dict) -> float:
    # create all possible orderings of tables
    permutations = list(itertools.permutations(tables))
    for entry in permutations:
        key = tuple(entry)

        if key in wj_data['card_ests_sum']:
            card_ests_sum = wj_data['card_ests_sum'][key]
            card_samples = wj_data['card_samples'][key]
            return card_ests_sum / card_samples

    # raise Exception(f"No card estimation found for tables {tables}\n{wj_data}")
    return 0


def get_wj_card_est(filter_conds: List[FilterCond], join_conds: List[JoinCond], duckdb_kwargs: Dict = None,
                    pg_kwargs: Dict = None):
    # run wanderjoin
    wj_data = run_wj_card_est(filter_conds=filter_conds, join_conds=join_conds, duckdb_kwargs=duckdb_kwargs,
                              pg_kwargs=pg_kwargs)

    # extract tables
    tables = list(
        set([f.table_name for f in filter_conds] + [j.table_name1 for j in join_conds] + [j.table_name2 for j in
                                                                                          join_conds]))

    # retrieve the card estimate
    return get_wj_card_for_table_comb(tables=tables, wj_data=wj_data)
