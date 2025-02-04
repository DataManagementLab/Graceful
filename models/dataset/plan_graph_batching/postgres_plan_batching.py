import collections
import copy
import os
import traceback
from typing import List, Dict, Optional

import xgboost
import dgl
import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from cross_db_benchmark.benchmark_tools.generate_workload import Operator
from flat_vector_baseline import extract_feats, encode_features
from models.preprocessing.feature_statistics import FeatureType
from models.zero_shot_models.depth_annotator import annotate_graph_with_depth_information
from models.zero_shot_models.specific_models.udf_edge_types import udf_node_types, udf_canonical_edge_types


def encode(column, plan_params, feature_statistics, est_card_udf_sel: Optional[int],
           card_type_below_udf: Optional[str] = None, card_type_above_udf: Optional[str] = None):
    assert column not in ['in_rows_act', 'in_rows_est', 'in_rows_deepdb', 'in_rows_wj']

    # make sure to keep the original name - this is needed to use the same min/max scaler whatever card est field is used
    # this allows to pretrain a model (e.g. on actual cardinalities) and during inference test different cardinality estimators.
    orig_name = column

    if column.endswith('_card') or column.endswith('_children_card'):
        assert card_type_below_udf is not None or card_type_above_udf is not None, f'{column} / {card_type_below_udf} / {card_type_above_udf}'

        # use a different cardinality column if we are above a UDF
        if plan_params['above_udf_filter'] or plan_params['is_udf_filter']:
            if column.endswith('_card') and not column.endswith('_children_card'):
                column = card_type_above_udf
            elif column.endswith('_children_card'):
                column = card_type_above_udf.replace('_card', '_children_card')
                assert column.endswith('_children_card')
            else:
                raise Exception(f'Unexpected case: {plan_params["above_udf_filter"]} / {plan_params["is_udf_filter"]}')

        elif not plan_params['above_udf_filter'] and not plan_params['is_udf_filter']:
            if column.endswith('_card') and not column.endswith('_children_card'):
                column = card_type_below_udf
            elif column.endswith('_children_card'):
                column = card_type_above_udf.replace('_card', '_children_card')
                assert column.endswith('_children_card')
            else:
                raise Exception(f'Unexpected case: {plan_params["above_udf_filter"]} / {plan_params["is_udf_filter"]}')
        else:
            raise Exception(f'Unexpected case: {plan_params["above_udf_filter"]} / {plan_params["is_udf_filter"]}')

    #  fallback in case actual cardinality is not in plan parameters
    if column == 'act_card' and column not in plan_params:
        value = 0
    else:
        value = plan_params[column]

    if column.endswith('_card') and not column.endswith('_children_card'):
        # apply assumed udf filter selectivity
        if (plan_params['above_udf_filter'] or plan_params['is_udf_filter']) and est_card_udf_sel is not None:
            value = value * est_card_udf_sel / 100
    elif column.endswith('_children_card'):
        # apply assumed udf filter selectivity
        if plan_params['above_udf_filter'] and not plan_params['is_udf_filter'] and est_card_udf_sel is not None:
            value = value * est_card_udf_sel / 100

    # encode value
    if feature_statistics[orig_name].get('type') == str(FeatureType.numeric):
        enc_value = feature_statistics[orig_name]['scaler'].transform(np.array([[value]])).item()
    elif feature_statistics[orig_name].get('type') == str(FeatureType.categorical):
        value_dict = feature_statistics[orig_name]['value_dict']
        try:
            enc_value = value_dict[str(value)]
        except KeyError as e:
            raise Exception(f"Could not find {value} in {value_dict} / {orig_name}")
    else:
        raise NotImplementedError
    return enc_value


def encode_udf(value, feat_name, feature_statistics):
    assert isinstance(value, int) or isinstance(value, float) or isinstance(value,
                                                                            str), f'{value} / {type(value)} / {feat_name} '

    if feature_statistics[feat_name].get('type') == str(FeatureType.numeric):
        try:
            enc_value = feature_statistics[feat_name]['scaler'].transform(np.array([[value]])).item()
        except Exception as e:
            raise Exception(f"Could not transform {value} in {feature_statistics[feat_name]['scaler']} / {feat_name}")
    elif feature_statistics[feat_name].get('type') == str(FeatureType.categorical):
        value_dict = feature_statistics[feat_name]['value_dict']
        try:
            enc_value = value_dict[value]
        except KeyError as e:
            raise Exception(f"Could not find {value} in {value_dict} / {feat_name}")
    elif feature_statistics[feat_name].get('type') == str(FeatureType.vector):
        # For the moment do nothing here; see later if we might need to do some stuff like scaling etc.
        enc_value = value
    else:
        raise NotImplementedError

    return enc_value


def plan_to_graph(node, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics, feature_statistics,
                  filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                  column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                  output_column_idx, column_idx, table_idx, plan_featurization, predicate_depths, intra_predicate_edges,
                  logical_preds, udf_internal_edges: Dict[str, List], udf_node_features: Dict, col_to_COMP_edges,
                  col_to_INVOC_edges, RET_to_outcol_edges, RET_to_filter_edges, udf_in_card_stats,
                  udf_filter_num_logicals_stats, udf_filter_num_literals_stats, plan_source_file: str, dbms: str,
                  multi_label_keep_duplicates: bool,
                  w_loop_end_node: bool, add_loop_loopend_edge: bool, est_card_udf_sel: int,
                  card_type_below_udf: Optional[str], card_type_above_udf: Optional[str],
                  card_type_in_udf: Optional[str],
                  flat_vector_feats_list: List,
                  parent_node_id=None,
                  depth=0, zs_paper_dataset: bool = False, card_est_assume_lazy_eval: bool = True):
    assert dbms is not None
    plan_node_id = len(plan_depths)
    plan_depths.append(depth)

    # add plan features
    plan_params = vars(node.plan_parameters)
    curr_plan_features = [encode(column, plan_params, feature_statistics, est_card_udf_sel=est_card_udf_sel,
                                 card_type_below_udf=card_type_below_udf,
                                 card_type_above_udf=card_type_above_udf) for column
                          in
                          plan_featurization['PLAN_FEATURES']]
    plan_features.append(curr_plan_features)

    # encode output columns which can in turn have several columns as a product in the aggregation
    output_columns = plan_params.get('output_columns')
    if output_columns is not None:
        for output_column in output_columns:

            # check if the output column is a "normal" output column or a UDF result
            if zs_paper_dataset or output_column.udf_output == "False":
                assert output_column.columns is not None, f'node: {node} / {output_column}'
                out_cols = tuple(output_column.columns)
                output_column_node_id = output_column_idx.get(
                    (output_column.aggregation, out_cols, database_id))

                # if not, create
                if output_column_node_id is None:
                    curr_output_column_features = [
                        encode(column, vars(output_column), feature_statistics, est_card_udf_sel=None)
                        for column in plan_featurization['OUTPUT_COLUMN_FEATURES']]

                    output_column_node_id = len(output_column_features)
                    output_column_features.append(curr_output_column_features)

                    # when searching for the output column id, we now also include information about the UDF name
                    # if no UDF name is used, then we will just have None as the UDF name
                    output_column_idx[(output_column.aggregation, tuple(output_column.columns), None, database_id)] \
                        = output_column_node_id

                    # featurize product of columns if there are any
                    db_column_features = db_statistics[database_id].column_stats
                    for column in output_column.columns:
                        column_node_id = column_idx.get((column, database_id))
                        if column_node_id is None:
                            curr_column_features = [
                                encode(feature_name, vars(db_column_features[column]), feature_statistics,
                                       est_card_udf_sel=None)
                                for feature_name in plan_featurization['COLUMN_FEATURES']]
                            column_node_id = len(column_features)
                            column_features.append(curr_column_features)
                            column_idx[(column, database_id)] = column_node_id
                        column_to_output_column_edges.append((column_node_id, output_column_node_id))

                # in any case add the corresponding edge
                output_column_to_plan_edges.append((output_column_node_id, plan_node_id))
            # if the output column is a UDF result
            else:
                curr_output_column_features = [
                    encode(column, vars(output_column), feature_statistics, est_card_udf_sel=None)
                    for column in plan_featurization['OUTPUT_COLUMN_FEATURES']]
                # since each UDF output is unique
                # create new output column node
                output_column_node_id = len(output_column_features)
                output_column_features.append(curr_output_column_features)

                # when searching for the output column id, we now also include information about the UDF name
                # if no UDF name is used, then we will just have None as the UDF name
                output_column_idx[(output_column.aggregation, output_column.columns, str(output_column.udf_name),
                                   database_id)] = output_column_node_id

                # featurize product of columns if there are any
                db_column_features = db_statistics[database_id].column_stats
                for param in plan_params.get('udf_params'):
                    column_node_id = column_idx.get((param, database_id))
                    if column_node_id is None:
                        curr_column_features = [
                            encode(feature_name, vars(db_column_features[param]), feature_statistics,
                                   est_card_udf_sel=None)
                            for feature_name in plan_featurization['COLUMN_FEATURES']]
                        column_node_id = len(column_features)
                        column_features.append(curr_column_features)
                        column_idx[(param, database_id)] = column_node_id
                    # column_to_output_column_edges.append((column_node_id, output_column_node_id))

                # call the helper function that creates the UDF graph => us output column_id as input parameter
                # link the UDF return nodes (src) to the output column node (dst)
                udf_to_graph(udf_name=output_column.udf_name, udf_internal_edges=udf_internal_edges,
                             udf_node_features=udf_node_features,
                             db_id=database_id, db_stats=db_statistics,
                             col_idx=column_idx, col_to_COMP_edges=col_to_COMP_edges,
                             col_to_INV_edges=col_to_INVOC_edges,
                             RET_to_outcol_edges=RET_to_outcol_edges,
                             RET_to_filter_edges=RET_to_filter_edges,
                             feature_statistics=feature_statistics, plan_featurization=plan_featurization,
                             table_id=plan_params["udf_table"],
                             filter_dst_id=None,
                             output_col_dst_id=output_column_node_id, plan_source_file=plan_source_file, dbms=dbms,
                             multi_label_keep_duplicates=multi_label_keep_duplicates,
                             udf_in_card_stats=udf_in_card_stats, w_loop_end_node=w_loop_end_node,
                             add_loop_loopend_edge=add_loop_loopend_edge, card_type_in_udf=card_type_in_udf,
                             card_est_assume_lazy_eval=card_est_assume_lazy_eval,
                             flat_vector_feats_list=flat_vector_feats_list)

                # link the output column node (src) to the corresponding plan node (dst)
                output_column_to_plan_edges.append((output_column_node_id, plan_node_id))

    # filter_columns (we do not reference the filter columns to columns since we anyway have to create a node per filter
    #  node)
    filter_column = plan_params.get('filter_columns')
    if filter_column is not None:
        # This list is a helper to keep track of the filter nodes
        # We will append the filter nodes to this list and then read the last element to which the udf graph is then linked to
        db_column_features = db_statistics[database_id].column_stats

        logical_preds_len_before = len(logical_preds)
        num_udf_ret_nodes_before = len(udf_node_features['RET'])

        parse_predicates(db_column_features, feature_statistics, filter_column, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                         logical_preds, plan_node_id=plan_node_id, db_statistics=db_statistics,
                         database_id=database_id, plan_params=plan_params, column_idx=column_idx,
                         column_features=column_features, udf_internal_edges=udf_internal_edges,
                         udf_node_features=udf_node_features, col_to_INVOC_edges=col_to_INVOC_edges,
                         RET_to_outcol_edges=RET_to_outcol_edges, RET_to_filter_edges=RET_to_filter_edges,
                         col_to_COMP_edges=col_to_COMP_edges,
                         plan_source_file=plan_source_file,
                         dbms=dbms, multi_label_keep_duplicates=multi_label_keep_duplicates,
                         udf_in_card_stats=udf_in_card_stats, w_loop_end_node=w_loop_end_node,
                         add_loop_loopend_edge=add_loop_loopend_edge,
                         card_est_assume_lazy_eval=card_est_assume_lazy_eval, card_type_in_udf=card_type_in_udf,
                         flat_vector_feats_list=flat_vector_feats_list)

        if len(udf_node_features['RET']) > num_udf_ret_nodes_before:
            # there is a UDF involved in the filter
            new_preds = logical_preds[logical_preds_len_before:]

            # calculate the number of logical predicates that were added
            new_logicals = sum(new_preds)
            new_literals = len(new_preds) - new_logicals

            udf_filter_num_logicals_stats.append(new_logicals)
            udf_filter_num_literals_stats.append(new_literals)

    # tables
    table = plan_params.get('table')
    if table is not None:
        table_node_id = table_idx.get((table, database_id))
        db_table_statistics = db_statistics[database_id].table_stats

        if table_node_id is None:
            curr_table_features = [
                encode(feature_name, vars(db_table_statistics[table]), feature_statistics, est_card_udf_sel=None)
                for feature_name in plan_featurization['TABLE_FEATURES']]
            table_node_id = len(table_features)
            table_features.append(curr_table_features)
            table_idx[(table, database_id)] = table_node_id

        table_to_plan_edges.append((table_node_id, plan_node_id))

    # add edge to parent
    if parent_node_id is not None:
        plan_to_plan_edges.append((plan_node_id, parent_node_id))

    # continue recursively
    for c in node.children:
        plan_to_graph(c, database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics,
                      feature_statistics,
                      filter_to_plan_edges, predicate_col_features, output_column_to_plan_edges, output_column_features,
                      column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                      output_column_idx, column_idx, table_idx, plan_featurization, predicate_depths,
                      intra_predicate_edges, logical_preds, udf_internal_edges=udf_internal_edges,
                      udf_node_features=udf_node_features,
                      col_to_COMP_edges=col_to_COMP_edges, col_to_INVOC_edges=col_to_INVOC_edges,
                      RET_to_outcol_edges=RET_to_outcol_edges, RET_to_filter_edges=RET_to_filter_edges,
                      parent_node_id=plan_node_id, depth=depth + 1, plan_source_file=plan_source_file, dbms=dbms,
                      multi_label_keep_duplicates=multi_label_keep_duplicates, zs_paper_dataset=zs_paper_dataset,
                      udf_in_card_stats=udf_in_card_stats, w_loop_end_node=w_loop_end_node,
                      add_loop_loopend_edge=add_loop_loopend_edge, card_est_assume_lazy_eval=card_est_assume_lazy_eval,
                      udf_filter_num_logicals_stats=udf_filter_num_logicals_stats,
                      udf_filter_num_literals_stats=udf_filter_num_literals_stats, est_card_udf_sel=est_card_udf_sel,
                      card_type_above_udf=card_type_above_udf, card_type_in_udf=card_type_in_udf,
                      card_type_below_udf=card_type_below_udf, flat_vector_feats_list=flat_vector_feats_list)


def parse_predicates(db_column_features, feature_statistics, filter_column, filter_to_plan_edges, plan_featurization,
                     predicate_col_features, predicate_depths, intra_predicate_edges, logical_preds,
                     db_statistics, database_id, plan_params, column_idx, column_features, udf_internal_edges,
                     udf_node_features, col_to_COMP_edges, col_to_INVOC_edges, RET_to_outcol_edges, RET_to_filter_edges,
                     plan_source_file: str,
                     dbms: str, multi_label_keep_duplicates: bool, udf_in_card_stats, card_type_in_udf: Optional[str],
                     flat_vector_feats_list: List,
                     plan_node_id=None,
                     parent_filter_node_id=None, depth=0, w_loop_end_node: bool = False,
                     add_loop_loopend_edge: bool = False, card_est_assume_lazy_eval: bool = True):
    """
    Recursive parsing of predicate columns

    :param db_column_features:
    :param feature_statistics:
    :param filter_column:
    :param filter_to_plan_edges:
    :param plan_featurization:
    :param plan_node_id:
    :param predicate_col_features:
    :return:
    """
    filter_node_id = len(predicate_depths)
    predicate_depths.append(depth)

    # gather features
    if filter_column.operator in {str(op) for op in list(Operator)}:
        filter_params = vars(filter_column)
        if hasattr(filter_column, "udf_name") and filter_column.udf_name is not None:
            filter_params['on_udf'] = True
        else:
            filter_params['on_udf'] = False
        curr_filter_features = [
            encode(feature_name, filter_params, feature_statistics, est_card_udf_sel=None, card_type_above_udf=None)
            for feature_name in plan_featurization['FILTER_FEATURES']]

        if filter_column.column is not None:
            curr_filter_col_feats = [
                encode(column, vars(db_column_features[filter_column.column]), feature_statistics,
                       est_card_udf_sel=None)
                for column in plan_featurization['COLUMN_FEATURES']]
        else:
            # hack for cases in which we have no base filter column (e.g., in a having clause where the column is some
            # result column of a subquery/groupby). In the future, this should be replaced by some graph model that also
            # encodes the structure of this output column.
            # this also includes predicates on UDFs
            curr_filter_col_feats = [0 for _ in plan_featurization['COLUMN_FEATURES']]
        curr_filter_features += curr_filter_col_feats
        logical_preds.append(False)

    else:
        filter_params = vars(filter_column)
        filter_params['on_udf'] = False
        curr_filter_features = [encode(feature_name, filter_params, feature_statistics, est_card_udf_sel=None)
                                for feature_name in plan_featurization['FILTER_FEATURES']]
        logical_preds.append(True)

    predicate_col_features.append(curr_filter_features)

    # add edge either to plan or inside predicates
    if depth == 0:
        assert plan_node_id is not None
        # in any case add the corresponding edge
        filter_to_plan_edges.append((filter_node_id, plan_node_id))
    else:
        assert parent_filter_node_id is not None
        intra_predicate_edges.append((filter_node_id, parent_filter_node_id))

    # recurse
    for c in filter_column.children:
        parse_predicates(db_column_features, feature_statistics, c, filter_to_plan_edges,
                         plan_featurization, predicate_col_features, predicate_depths, intra_predicate_edges,
                         logical_preds, parent_filter_node_id=filter_node_id, depth=depth + 1,
                         db_statistics=db_statistics,
                         database_id=database_id, plan_params=plan_params, column_idx=column_idx,
                         column_features=column_features, udf_internal_edges=udf_internal_edges,
                         udf_node_features=udf_node_features, col_to_INVOC_edges=col_to_INVOC_edges,
                         col_to_COMP_edges=col_to_COMP_edges,
                         RET_to_outcol_edges=RET_to_outcol_edges, RET_to_filter_edges=RET_to_filter_edges,
                         plan_source_file=plan_source_file,
                         dbms=dbms, multi_label_keep_duplicates=multi_label_keep_duplicates,
                         udf_in_card_stats=udf_in_card_stats, w_loop_end_node=w_loop_end_node,
                         add_loop_loopend_edge=add_loop_loopend_edge,
                         card_est_assume_lazy_eval=card_est_assume_lazy_eval, card_type_in_udf=card_type_in_udf,
                         flat_vector_feats_list=flat_vector_feats_list)

    # filter_column has one udf specific attribute => udf_name
    # if udf_name is None, then no UDF is used
    # if udf_name is not None, than we have to create a UDF graph
    # if there is a UDF involved in the filter, then we have to create a UDF graph
    # and establish the connections between the UDF graph and the filter operator that was just added
    if hasattr(filter_column, "udf_name") and filter_column.udf_name is not None:
        assert filter_column.children is None or len(
            filter_column.children) == 0, "UDF filter column expected to have no children when seeing an UDF"
        # make sure that all columns of the udf are added to the column features
        db_column_features = db_statistics[database_id].column_stats
        assert plan_params.get('udf_params') is not None, f'{plan_params}'

        for param in plan_params.get('udf_params'):
            column_node_id = column_idx.get((param, database_id))
            if column_node_id is None:
                curr_column_features = [
                    encode(feature_name, vars(db_column_features[param]), feature_statistics, est_card_udf_sel=None)
                    for feature_name in plan_featurization['COLUMN_FEATURES']]
                column_node_id = len(column_features)
                column_features.append(curr_column_features)
                column_idx[(param, database_id)] = column_node_id

        udf_to_graph(udf_name=filter_column.udf_name, db_id=database_id, db_stats=db_statistics,
                     col_idx=column_idx, col_to_COMP_edges=col_to_COMP_edges, col_to_INV_edges=col_to_INVOC_edges,
                     feature_statistics=feature_statistics, plan_featurization=plan_featurization,
                     table_id=plan_params["udf_table"], udf_internal_edges=udf_internal_edges,
                     udf_node_features=udf_node_features, RET_to_outcol_edges=RET_to_outcol_edges,
                     RET_to_filter_edges=RET_to_filter_edges, filter_dst_id=filter_node_id,
                     output_col_dst_id=None, plan_source_file=plan_source_file, dbms=dbms,
                     multi_label_keep_duplicates=multi_label_keep_duplicates, udf_in_card_stats=udf_in_card_stats,
                     w_loop_end_node=w_loop_end_node, add_loop_loopend_edge=add_loop_loopend_edge,
                     card_est_assume_lazy_eval=card_est_assume_lazy_eval, card_type_in_udf=card_type_in_udf,
                     flat_vector_feats_list=flat_vector_feats_list)


lookup = {
    'RETURN': 'RET',
    'COMP': 'COMP',
    'BRANCH': 'BRANCH',
    'LOOP_HEAD': 'LOOP',
    'LOOP_END': 'LOOPEND',
    'INVOCATION': 'INV'
}


def udf_to_graph(udf_name, w_loop_end_node: bool, udf_internal_edges, udf_node_features, db_id, col_idx,
                 col_to_COMP_edges,
                 col_to_INV_edges, RET_to_outcol_edges, RET_to_filter_edges, db_stats, udf_in_card_stats,
                 feature_statistics, plan_featurization, table_id: int, filter_dst_id: Optional[int],
                 output_col_dst_id: Optional[int], card_type_in_udf: Optional[str],
                 plan_source_file: str, dbms: str, multi_label_keep_duplicates=False,
                 add_loop_loopend_edge: bool = False, card_est_assume_lazy_eval: bool = True,
                 flat_vector_feats_list: List = None):
    atts = vars(db_stats[db_id])

    udf_suffix = []
    if w_loop_end_node:
        udf_suffix.append('loopend')
    if card_est_assume_lazy_eval:
        udf_suffix.append('lazy')

    if len(udf_suffix) > 0:
        udf_suffix = '_'.join(udf_suffix)
        udf_suffix = f'.{udf_suffix}'
    else:
        udf_suffix = ''

    udf_path = os.path.dirname(os.path.dirname(os.path.dirname(plan_source_file)))

    full_path = os.path.join(udf_path, 'dbs', atts.get("database_name"), 'created_graphs',
                             udf_name + f"{udf_suffix}.gpickle")
    assert os.path.exists(full_path), full_path
    udf_graph: nx.DiGraph = nx.read_gpickle(full_path)  # restore UDF graph from gpickle
    col_edges = []
    nx_dgl_map = {}

    if flat_vector_feats_list is not None:
        flat_vector_feats_list.append(extract_feats(udf_graph))

    for edge in udf_graph.edges:
        src = edge[0]
        dst = edge[1]

        if udf_graph.nodes[src]["type"] == "VAR":
            col_edges.append(edge)
        else:
            if udf_graph.nodes[src]['type'] == 'INVOCATION':
                # extract statistics
                udf_in_card_stats.append(udf_graph.nodes[src]['in_rows_act'])

            # src and dst both do not exist yet
            if (udf_graph.nodes[src]["type"], udf_graph.nodes[dst]["type"]) in [('INVOCATION', 'COMP'),
                                                                                ('INVOCATION', 'BRANCH'),
                                                                                ('INVOCATION', 'LOOP_HEAD'),
                                                                                ('INVOCATION', 'RETURN'), ]:
                # translate node types
                orig_src_name = udf_graph.nodes[src]['type']
                orig_dst_name = udf_graph.nodes[dst]['type']
                src_type = lookup[orig_src_name]
                dst_type = lookup[orig_dst_name]

                # udf_internal_edges[f'{src_type}_{dst_type}'].append(
                #     (len(udf_node_features[src_type]), len(udf_node_features[dst_type])))

                assert src not in nx_dgl_map
                nx_dgl_map[src] = len(udf_node_features[src_type])
                assert dst not in nx_dgl_map
                nx_dgl_map[dst] = len(udf_node_features[dst_type])
                # retrieve the necessary features from INV part
                feat_lst = create_udf_feat_list(feature_statistics, plan_featurization[f'{src_type}_FEATURES'],
                                                src, udf_graph, card_type_in_udf=card_type_in_udf,
                                                multi_label_keep_duplicates=multi_label_keep_duplicates)
                udf_node_features[src_type].append(feat_lst)
                # retrieve the necessary features from COMP part
                feat_lst = create_udf_feat_list(feature_statistics, plan_featurization[f'{dst_type}_FEATURES'],
                                                dst, udf_graph, card_type_in_udf=card_type_in_udf,
                                                multi_label_keep_duplicates=multi_label_keep_duplicates)
                udf_node_features[dst_type].append(feat_lst)

                udf_internal_edges[f'{src_type}_{dst_type}'].append((nx_dgl_map[src], nx_dgl_map[dst]))

            # src exists and dst might do as well
            elif (udf_graph.nodes[src]["type"], udf_graph.nodes[dst]["type"]) in [('COMP', 'RETURN'),
                                                                                  ('LOOP_HEAD', 'RETURN'),
                                                                                  ('LOOP_END', 'RETURN'),
                                                                                  ('COMP', 'BRANCH'),
                                                                                  ('LOOP_END', 'BRANCH'),
                                                                                  ('COMP', 'LOOP_HEAD'),
                                                                                  ('LOOP_END', 'LOOP_HEAD'),
                                                                                  ('COMP', 'COMP'),
                                                                                  ('LOOP_END', 'COMP'),
                                                                                  ('COMP', 'LOOP_END'),
                                                                                  ('LOOP_END', 'LOOP_END')
                                                                                  ]:
                # translate node types
                orig_dst_name = udf_graph.nodes[dst]['type']
                src_type = lookup[udf_graph.nodes[src]['type']]
                dst_type = lookup[orig_dst_name]

                assert src in nx_dgl_map, f'{src} / {nx_dgl_map}'
                if dst not in nx_dgl_map.keys():
                    nx_dgl_map[dst] = len(udf_node_features[dst_type])
                    # here only consider the dst for featurization since the src is already covered in a different edge
                    # retrieve the necessary features from COMP part
                    feat_lst = create_udf_feat_list(feature_statistics, plan_featurization[f'{dst_type}_FEATURES'],
                                                    dst, udf_graph, card_type_in_udf=card_type_in_udf,
                                                    multi_label_keep_duplicates=multi_label_keep_duplicates)
                    udf_node_features[dst_type].append(feat_lst)

                udf_internal_edges[f'{src_type}_{dst_type}'].append((nx_dgl_map[src], nx_dgl_map[dst]))

            # src exists and dst not
            elif (udf_graph.nodes[src]["type"], udf_graph.nodes[dst]["type"]) in [('BRANCH', 'BRANCH'),
                                                                                  ('BRANCH', 'COMP'),
                                                                                  ('BRANCH', 'LOOP_HEAD'),
                                                                                  ('LOOP_HEAD', 'COMP'),
                                                                                  ('LOOP_HEAD', 'BRANCH'),
                                                                                  ('LOOP_HEAD', 'LOOP_HEAD'),
                                                                                  ('LOOP_HEAD', 'LOOP_END'), ]:
                # translate node types
                orig_dst_name = udf_graph.nodes[dst]['type']
                src_type = lookup[udf_graph.nodes[src]['type']]
                dst_type = lookup[orig_dst_name]

                # udf_internal_edges[f'{src_type}_{dst_type}'].append(
                #     (nx_dgl_map[src], len(udf_node_features[dst_type])))

                assert dst not in nx_dgl_map
                nx_dgl_map[dst] = len(udf_node_features[dst_type])
                # here only consider the dst for featurization since the src is already covered in a different edge
                # retrieve the necessary features from COMP part
                feat_lst = create_udf_feat_list(feature_statistics, plan_featurization[f'{dst_type}_FEATURES'],
                                                dst,
                                                udf_graph, card_type_in_udf=card_type_in_udf,
                                                multi_label_keep_duplicates=multi_label_keep_duplicates)
                udf_node_features[dst_type].append(feat_lst)

                skip_loop_loopend_edge = False
                if not add_loop_loopend_edge:
                    # remove the edge between loop->loopend (shortcutting the loop body)
                    if (src_type, dst_type) == ('LOOP_HEAD', 'LOOP_END'):
                        # check that there is an alternative route through the body / i.e. check that body exists (two output edges)
                        loop_head_successors = udf_graph.successors(src)
                        succ_types = [udf_graph.nodes[succ]['type'] for succ in loop_head_successors]
                        assert 'LOOP_END' in succ_types, f'{loop_head_successors} / {udf_graph.nodes}'
                        if len(loop_head_successors) > 1:
                            assert 'COMP' in succ_types or 'BRANCH' in succ_types, f'{loop_head_successors} / {udf_graph.nodes}'

                            # skip the edge between loop->loopend
                            skip_loop_loopend_edge = True

                if not skip_loop_loopend_edge:
                    udf_internal_edges[f'{src_type}_{dst_type}'].append((nx_dgl_map[src], nx_dgl_map[dst]))
            else:
                raise Exception(f"Unknown edge type: {udf_graph.nodes[src]['type']} -> {udf_graph.nodes[dst]['type']}")

    # add edges from col to COMP, col to INVOCATION, and col to BRANCH
    for edge in col_edges:
        src = edge[0]
        dst = edge[1]
        var_id = udf_graph.nodes[src]["var_id"]

        c = get_col_id(var_id, db_id, table_id, db_stats, dbms=dbms)
        assert (c, db_id) in col_idx.keys(), f'({c},{db_id}) not in {col_idx.keys()}'
        col_id = col_idx[(c, db_id)]

        if udf_graph.nodes[dst]["type"] == "COMP":
            col_to_COMP_edges.append((col_id, nx_dgl_map[dst]))

        elif udf_graph.nodes[dst]["type"] == "INVOCATION":
            try:
                col_to_INV_edges.append((col_id, nx_dgl_map[dst]))
            except KeyError as e:
                print(f'{dst} / COL->INV: {len(col_to_INV_edges)} / {nx_dgl_map}')
                raise e
        else:
            raise Exception(f"Unknown edge type: {udf_graph.nodes[src]['type']} -> {udf_graph.nodes[dst]['type']}")

    # Finally, add an edge from return to an output column or filter node
    # If the udf is part of a filter, then we will link to a filter node
    # If the udf result is aggregated (e.g., SUM(func(...)) or if the udf is used for group by, then
    # then the udf is linked to an output_column node
    if output_col_dst_id is not None:
        assert filter_dst_id is None
        RET_to_outcol_edges.append((len(udf_node_features['RET']) - 1, output_col_dst_id))
    else:
        assert filter_dst_id is not None
        RET_to_filter_edges.append((len(udf_node_features['RET']) - 1, filter_dst_id))


def create_udf_feat_list(feature_statistics, plan_featurization, src, udf_graph, card_type_in_udf: Optional[str],
                         multi_label=["in_dts", "ops", "cmdtypes"],
                         multi_label_keep_duplicates: bool = False):
    feat_lst = []
    for feat in plan_featurization:

        # decide whether to use a different cardinality estimator - but still use the original scaling configuration
        if feat.startswith('in_rows_'):
            assert card_type_in_udf in ['dd', 'wj', 'act', 'est'], f'{feat} / {card_type_in_udf}'
            if card_type_in_udf == 'dd':
                feat_name = 'in_rows_deepdb'
            else:
                feat_name = f'in_rows_{card_type_in_udf}'
        else:
            feat_name = feat

        assert feat_name in udf_graph.nodes[src], f'{feat_name} not in {udf_graph.nodes[src]}'

        value = udf_graph.nodes[src][feat_name]

        if feat in multi_label:
            target_list_len = max(10, feature_statistics[feat]['no_vals'])

            if not multi_label_keep_duplicates:
                values = list(set(value))  # only keep unique elements
            else:
                values = value

                if len(values) > target_list_len:
                    values = values[:target_list_len]
            intermed_lst = []
            for val in values:
                intermed_lst.append(encode_udf(val, feat, feature_statistics))

            for i in range(0, target_list_len - len(intermed_lst)):
                intermed_lst.append(-1)
            feat_lst += intermed_lst

        else:
            if feat == "lib":  # handle the large lib vector differently
                feat_lst += list(value)
            else:
                feat_lst.append(encode_udf(value, feat, feature_statistics))
    return feat_lst


def get_col_id(var_id, db_id: int, table_id: int, db_stats, dbms: str):
    # retrieve table name as string from table_statistsics
    assert isinstance(db_id, int), f'{db_id} is not an integer: {type(db_id)}'
    assert isinstance(table_id, int), f'{table_id} is not an integer: {type(table_id)}'
    atts = vars(db_stats[db_id].table_stats[table_id])

    if dbms == 'postgres':
        table_name = atts.get("relname")
        attname_str = 'attname'
        tablename_str = 'tablename'
    elif dbms == 'duckdb':
        table_name = atts.get("table_name")
        attname_str = 'column_name'
        tablename_str = 'table_name'
    else:
        raise Exception(f'Unknown DBMS: {dbms}')

    assert table_name is not None, f'{atts} \n {dbms}'

    for idx, elem in enumerate(db_stats[db_id].column_stats):
        atts = vars(elem)
        if atts.get(attname_str) == var_id and atts.get(tablename_str) == table_name:
            return idx

    raise Exception(f"Could not find column {var_id} in table {table_name} \n {db_stats}")


def find_udf_node(plan):
    # extract operator that involves the udf from plan
    if hasattr(plan.plan_parameters, 'udf_name') or hasattr(plan.plan_parameters, 'udf_params') or hasattr(
            plan.plan_parameters, 'udf_table'):
        return plan
    else:
        # check child nodes
        for c in plan.children:
            res = find_udf_node(c)
            if res is not None:
                return res
        return None


def prune_udf_information_from_filter(filter):
    # remove any udf related information from the filter
    if hasattr(filter, 'udf_name'):
        delattr(filter, 'udf_name')
    if hasattr(filter, 'udf_params'):
        delattr(filter, 'udf_params')
    if hasattr(filter, 'udf_table'):
        delattr(filter, 'udf_table')

    if hasattr(filter, 'children'):
        for c in filter.children:
            prune_udf_information_from_filter(c)


def prune_udf_information(plan):
    # remove any udf related informaiton from the query plan
    if hasattr(plan.plan_parameters, 'udf_name'):
        delattr(plan.plan_parameters, 'udf_name')
    if hasattr(plan.plan_parameters, 'udf_params'):
        delattr(plan.plan_parameters, 'udf_params')
    if hasattr(plan.plan_parameters, 'udf_table'):
        delattr(plan.plan_parameters, 'udf_table')

    if hasattr(plan.plan_parameters, 'output_columns'):
        for col in plan.plan_parameters.output_columns:
            col.udf_output = 'False'
            if hasattr(col, 'udf_name'):
                delattr(col, 'udf_name')
                col.columns = []  # adding here a real column could improve results

            assert col.columns is not None, f'plan: {plan}'

    if hasattr(plan.plan_parameters, 'filter_columns'):
        prune_udf_information_from_filter(plan.plan_parameters.filter_columns)

    if hasattr(plan, 'udf'):
        delattr(plan, 'udf')
    if hasattr(plan, 'udf_pullup'):
        plan.udf_pullup = False

    plan.plan_parameters.above_udf_filter = False
    plan.plan_parameters.is_udf_filter = False

    # iterate over children
    for c in plan.children:
        prune_udf_information(c)


def add_pseudo_table_to_stats(db_statistics: Dict, database_id: int, pseudo_table_name: int, num_rows: int) -> int:
    # add information about the input to the UDF to the table_stats
    assert database_id in db_statistics
    assert pseudo_table_name not in db_statistics[database_id].table_stats
    db_statistics[database_id].table_stats.append({'table_name': pseudo_table_name,
                                                   'estimated_size': num_rows})
    return len(db_statistics[database_id].table_stats) - 1


def split_graph_into_udf_and_sql(plan, db_statistics: Dict):
    # split the graph into the UDF graph (a reduced query plan with only the UDF,
    # and the plan of the surrounding query (with all UDF information removed).

    # find the root node of the udf
    orig_udf_node = find_udf_node(plan)

    assert orig_udf_node is not None, f'Could not find UDF node in plan: {plan}'

    # create a copy of the UDF node element
    udf_plan = copy.deepcopy(orig_udf_node)

    # extarct udf_name
    udf_name = plan.udf.udf_name

    if len(udf_plan.children) == 0:
        # no children => no changes to UDF graph necessary
        pass
    elif len(udf_plan.children) == 1:
        # introduce a virtual scan child node
        child = udf_plan.children[0]
        if len(child.children) == 0:
            # child has no children. I.e. do nothing. It is already a scan / ...
            pass
        else:
            # overwrite the child node
            child.children = []

            # delete attr
            attr_to_delete = ['join', 'filter_columns', 'literal_feature', 'text']
            for attr in attr_to_delete:
                if hasattr(child.plan_parameters, attr):
                    delattr(child.plan_parameters, attr)

            # overwrite operator
            child.plan_parameters.op_name = 'SEQ_SCAN'
            child.plan_parameters.table_name = udf_name

            # create pseudo table
            pseudo_table_id = add_pseudo_table_to_stats(db_statistics, database_id=plan.database_id,
                                                        pseudo_table_name=udf_name,
                                                        num_rows=child.plan_parameters.act_card)
            child.plan_parameters.table_id = pseudo_table_id

            # overwrite child card
            child.act_children_card = 1
            child.est_children_card = 1
            child.dd_est_children_card = 1
            child.wj_est_children_card = 1

    else:
        raise Exception(f'UDF node has more than one child: {udf_plan.children}')

    # copy over root node info
    udf_plan.query = plan.query
    udf_plan.plan_runtime_ms = -42
    udf_plan.num_tables = 1
    udf_plan.num_filters = 0
    udf_plan.database_name = plan.database_name
    udf_plan.database_id = plan.database_id
    udf_plan.source_file = plan.source_file
    udf_plan.udf_pullup = plan.udf_pullup
    udf_plan.udf = plan.udf

    # from sql_plan, prune all udf information
    prune_udf_information(plan)

    return plan, udf_plan


def split_graphs_into_udf_and_sql(plans, db_statistics: Dict):
    # for ablation study: udf_cost + sql_cost = total_cost
    out_plans = []
    udf_graph_bitmask = []
    for sample_idx, plan in plans:
        # check if the plan contains a UDF
        if hasattr(plan, 'udf') and plan.udf is not None:
            # do the split
            sql_plan, udf_plan = split_graph_into_udf_and_sql(plan, db_statistics)
            # add both plans
            out_plans.append((sample_idx, sql_plan))
            out_plans.append((sample_idx, udf_plan))
            udf_graph_bitmask.append(False)
            udf_graph_bitmask.append(True)
        else:
            out_plans.append((sample_idx, plan))
            udf_graph_bitmask.append(False)

    return out_plans, udf_graph_bitmask


def postgres_plan_collator(plans, est_card_udf_sel: Optional[int],
                           card_type_below_udf: Optional[str], card_type_above_udf: Optional[str],
                           card_type_in_udf: Optional[str], feature_statistics=None, db_statistics=None,
                           featurization=None, dbms: str = None,
                           offset_np_import: int = 0, multi_label_keep_duplicates: bool = False,
                           zs_paper_dataset: bool = False, train_udf_graph_against_udf_runtime: bool = False,
                           w_loop_end_node: bool = False, add_loop_loopend_edge: bool = False,
                           card_est_assume_lazy_eval: bool = True, plans_have_no_udf: bool = False,
                           skip_udf: bool = False, separate_sql_udf_graphs: bool = False,
                           annotate_flat_vector_udf_preds:bool=False, flat_vector_model_path:str=None
                           ):
    """
    Combines physical plans into a large graph that can be fed into ML models.
    :return:
    """
    try:
        if separate_sql_udf_graphs or annotate_flat_vector_udf_preds:
            plans, udf_graph_bitmask = split_graphs_into_udf_and_sql(plans, db_statistics=db_statistics)
        else:
            udf_graph_bitmask = None

        assert dbms is not None

        # output:
        #   - list of labels (i.e., plan runtimes)
        #   - feature dictionaries
        #       - column_features: matrix
        #       - output_column_features: matrix
        #       - filter_column_features: matrix
        #       - plan_node_features: matrix
        #       - table_features: matrix
        #       - logical_pred_features: matrix
        #   - edges
        #       - table_to_output
        #       - column_to_output
        #       - filter_to_plan
        #       - output_to_plan
        #       - plan_to_plan
        #       - intra predicate (e.g., column to AND)
        plan_depths = []
        plan_features = []
        plan_to_plan_edges = []
        filter_to_plan_edges = []
        filter_features = []
        output_column_to_plan_edges = []
        output_column_features = []
        column_to_output_column_edges = []
        column_features = []
        table_features = []
        table_to_plan_edges = []
        labels_ms = []
        predicate_depths = []
        intra_predicate_edges = []
        logical_preds = []

        # Added for UDF support
        # Edges to connect the UDF with the "outer" graph
        RET_to_outcol_edges = []
        RET_to_filter_edges = []
        col_to_INVOC_edges = []
        col_to_COMP_edges = []

        # Edges to connect the different nodes within a UDF
        udf_internal_edges = dict(
            INV_COMP=[],
            COMP_COMP=[],
            COMP_RET=[],
            COMP_BRANCH=[],
            BRANCH_COMP=[],
            BRANCH_BRANCH=[],
            INV_BRANCH=[],
            LOOP_COMP=[],
            COMP_LOOP=[],
            LOOPEND_COMP=[],
            COMP_LOOPEND=[],
            BRANCH_LOOP=[],
            LOOP_BRANCH=[],
            LOOPEND_BRANCH=[],
            LOOP_RET=[],
            LOOPEND_RET=[],
            INV_LOOP=[],
            INV_RET=[],
            LOOP_LOOP=[],
            LOOPEND_LOOP=[],
            LOOP_LOOPEND=[],
            LOOPEND_LOOPEND=[]
        )
        udf_node_features = dict(
            INV=[],
            COMP=[],
            RET=[],
            BRANCH=[],
            LOOP=[],
            LOOPEND=[],
        )

        output_column_idx = dict()
        column_idx = dict()
        table_idx = dict()

        # plan stats
        num_joins = []
        num_filters = []
        udf_num_np_calls = []
        udf_num_math_calls = []
        udf_num_comp_nodes = []
        udf_num_branches = []
        udf_num_loops = []
        udf_in_card = []
        sql_list = []
        udf_pos_in_query = []
        database_name_list = []
        udf_filter_num_literals = []
        udf_filter_num_logicals = []
        graph_num_nodes = []
        graph_num_edges = []

        # prepare robust encoder for the numerical fields
        add_numerical_scalers(feature_statistics)

        # prepare flat vector features list - only used for baseline: udf-cost with flat-vector, sql cost with our model
        if annotate_flat_vector_udf_preds:
            flat_vector_feats_list = []
        else:
            flat_vector_feats_list = None

        def get_num_nodes_edges():
            """
            Helper function to get the number of nodes and edges in the graph
            """
            num_nodes = 0
            num_edges = 0
            for k, v in udf_internal_edges.items():
                num_edges += len(v)
            for k, v in udf_node_features.items():
                num_nodes += len(v)

            # manually add edges for the UDF nodes
            num_edges += len(RET_to_outcol_edges)
            num_edges += len(RET_to_filter_edges)
            num_edges += len(col_to_INVOC_edges)
            num_edges += len(col_to_COMP_edges)

            # add the number of nodes for the non-udf graph
            num_nodes += len(plan_features)
            num_nodes += len(filter_features)
            num_nodes += len(output_column_features)
            num_nodes += len(column_features)
            num_nodes += len(table_features)

            # add the number of edges for the non-udf graph
            num_edges += len(plan_to_plan_edges)
            num_edges += len(filter_to_plan_edges)
            num_edges += len(output_column_to_plan_edges)
            num_edges += len(column_to_output_column_edges)
            num_edges += len(table_to_plan_edges)
            num_edges += len(intra_predicate_edges)
            return num_nodes, num_edges

        # iterate over plans and create lists of edges and features per node
        sample_idxs = []
        for sample_idx, p in plans:
            # store number of nodes before parsing this plan (due to batching they will all result in a single graph)
            prev_iter_num_comp_nodes = len(udf_node_features['COMP'])
            prev_iter_num_loop_nodes = len(udf_node_features['LOOP'])
            prev_iter_num_branch_nodes = len(udf_node_features['BRANCH'])
            prev_iter_udf_in_card = len(udf_in_card)

            # store number of nodes and edges before parsing this plan (due to batching they will all result in a single graph)
            prev_num_nodes, prev_num_edges = get_num_nodes_edges()

            sample_idxs.append(sample_idx)

            ret_feats_before = len(udf_node_features['RET'])
            try:
                plan_to_graph(p, p.database_id, plan_depths, plan_features, plan_to_plan_edges, db_statistics,
                              feature_statistics,
                              filter_to_plan_edges, filter_features, output_column_to_plan_edges,
                              output_column_features,
                              column_to_output_column_edges, column_features, table_features, table_to_plan_edges,
                              output_column_idx, column_idx, table_idx, featurization, predicate_depths,
                              intra_predicate_edges, logical_preds,
                              udf_internal_edges=udf_internal_edges,
                              udf_node_features=udf_node_features,
                              col_to_COMP_edges=col_to_COMP_edges,
                              plan_source_file=p.source_file,
                              col_to_INVOC_edges=col_to_INVOC_edges,
                              RET_to_outcol_edges=RET_to_outcol_edges, RET_to_filter_edges=RET_to_filter_edges,
                              dbms=dbms,
                              multi_label_keep_duplicates=multi_label_keep_duplicates,
                              zs_paper_dataset=zs_paper_dataset,
                              udf_in_card_stats=udf_in_card, w_loop_end_node=w_loop_end_node,
                              add_loop_loopend_edge=add_loop_loopend_edge,
                              card_est_assume_lazy_eval=card_est_assume_lazy_eval,
                              udf_filter_num_logicals_stats=udf_filter_num_logicals,
                              udf_filter_num_literals_stats=udf_filter_num_literals, est_card_udf_sel=est_card_udf_sel,
                              card_type_below_udf=card_type_below_udf,
                              card_type_above_udf=card_type_above_udf, card_type_in_udf=card_type_in_udf,flat_vector_feats_list=flat_vector_feats_list)
            except Exception as e:
                print(
                    f'Card type below: {card_type_below_udf}, card type above: {card_type_above_udf}, card type in udf: {card_type_in_udf}',
                    flush=True)
                print(p, flush=True)
                raise e
                # raise Exception(f'{e}\n{p}')

            num_ret_nodes = len(udf_node_features['RET'])
            if not zs_paper_dataset and hasattr(p, 'udf'):
                assert num_ret_nodes == ret_feats_before + 1, f'{num_ret_nodes} != {ret_feats_before}+1\n{p}'
                assert len(
                    udf_in_card) == prev_iter_udf_in_card + 1, f'{len(udf_in_card)} != {prev_iter_udf_in_card} + 1'
            else:
                assert num_ret_nodes == ret_feats_before, f'{num_ret_nodes} != {ret_feats_before}\n{p}'

            if not zs_paper_dataset:
                runtime_ms = p.plan_runtime_ms
                if offset_np_import is not None and offset_np_import != 0 and hasattr(p,
                                                                                      'udf') and p.udf.udf_numpy_lib_imported:
                    # add offset to the numpy import
                    runtime_ms -= offset_np_import
            else:
                runtime_ms = p.plan_runtime

            if train_udf_graph_against_udf_runtime:
                udf_timing_s = p.udf_timing
                labels_ms.append(udf_timing_s * 1000)
            else:
                labels_ms.append(runtime_ms)

            if zs_paper_dataset:
                udf_num_np_calls.append(0)
                udf_num_math_calls.append(0)
                num_joins.append(-1)
                num_filters.append(-1)
                udf_pos_in_query.append('none')
                udf_in_card.append(-1)
            else:
                num_joins.append(p.num_tables - 1)
                num_filters.append(p.num_filters)
                if hasattr(p, 'udf'):
                    udf_num_np_calls.append(p.udf.udf_num_np_calls)
                    udf_num_math_calls.append(p.udf.udf_num_math_calls)
                    if p.udf_pullup:
                        udf_pos_in_query.append(f'{p.udf.udf_pos_in_query}_pullup')
                    else:
                        udf_pos_in_query.append(p.udf.udf_pos_in_query)
                else:
                    udf_num_np_calls.append(0)
                    udf_num_math_calls.append(0)
                    udf_pos_in_query.append('none')
                    udf_in_card.append(-1)  # udf card is otherwise annotated in the udf_to_graph function
            udf_num_comp_nodes.append(len(udf_node_features['COMP']) - prev_iter_num_comp_nodes)
            udf_num_branches.append(len(udf_node_features['BRANCH']) - prev_iter_num_branch_nodes)
            udf_num_loops.append(len(udf_node_features['LOOP']) - prev_iter_num_loop_nodes)
            sql_list.append(p.query)
            database_name_list.append(p.database_name)

            # compute number of nodes and edges after parsing this plan (due to batching they will all result in a single graph)
            after_num_nodes, after_num_edges = get_num_nodes_edges()
            graph_num_nodes.append(after_num_nodes - prev_num_nodes)
            graph_num_edges.append(after_num_edges - prev_num_edges)

            assert len(udf_filter_num_logicals) == len(
                udf_filter_num_literals), f'{len(udf_filter_num_logicals)} != {len(udf_filter_num_literals)}'
            if hasattr(p, 'udf') and p.udf.udf_pos_in_query == 'filter':
                assert len(udf_filter_num_logicals) == len(
                    udf_pos_in_query), f'{len(udf_filter_num_logicals)} != {len(udf_pos_in_query)}'
            else:
                udf_filter_num_literals.append(-1)
                udf_filter_num_logicals.append(-1)

        assert len(udf_in_card) == len(udf_pos_in_query), f'{len(udf_in_card)} != {len(udf_pos_in_query)}'

        if skip_udf:
            for n in ['INV', 'COMP', 'BRANCH', 'LOOP', 'LOOPEND']:
                udf_node_features[n] = []

            col_to_INVOC_edges = []
            col_to_COMP_edges = []

            for key in udf_internal_edges:
                udf_internal_edges[key] = []

        stats = dict(
            num_joins=num_joins,
            num_filters=num_filters,
            udf_num_np_calls=udf_num_np_calls,
            udf_num_math_calls=udf_num_math_calls,
            udf_num_comp_nodes=udf_num_comp_nodes,
            udf_num_branches=udf_num_branches,
            udf_num_loops=udf_num_loops,
            sql_list=sql_list,
            udf_pos_in_query=udf_pos_in_query,
            udf_in_card=udf_in_card,
            database_name=database_name_list,
            udf_filter_num_logicals=udf_filter_num_logicals,
            udf_filter_num_literals=udf_filter_num_literals,
            graph_num_nodes=graph_num_nodes,
            graph_num_edges=graph_num_edges,
        )

        # run flat vector model
        if flat_vector_feats_list is not None:
            if len(flat_vector_feats_list)>0:
                # compute flat vector predictions
                model = xgboost.XGBRegressor()
                assert flat_vector_model_path is not None
                model.load_model(flat_vector_model_path)

                encoded_flat_vector_feats_list = encode_features(flat_vector_feats_list,onehot_all_ops=False,onehot_np_ops=True)

                flat_vector_preds = model.predict(encoded_flat_vector_feats_list)

                # extract number of udf invocations and multiply with per tuple cost estimates
                act_udf_invoc = [card for card in stats['udf_in_card'] if card != -1]
                flat_vector_preds = [pred * card for pred, card in zip(flat_vector_preds, act_udf_invoc)]

                stats['flat_vector_predictions']=flat_vector_preds
            else:
                stats['flat_vector_predictions'] = []

        assert len(labels_ms) == len(plans)
        assert len(plan_depths) == len(plan_features)

        if not train_udf_graph_against_udf_runtime:
            data_dict, nodes_per_depth, plan_dict = create_node_types_per_depth(plan_depths, plan_to_plan_edges)
        else:
            data_dict = dict()
            nodes_per_depth = dict()
            plan_dict = dict()

        # similarly create node types:
        pred_dict = dict()
        nodes_per_pred_depth = collections.defaultdict(int)
        no_filter_columns = 0
        for pred_node, d in enumerate(predicate_depths):
            # predicate node
            if logical_preds[pred_node]:
                pred_dict[pred_node] = (nodes_per_pred_depth[d], d)
                nodes_per_pred_depth[d] += 1
            # filter column
            else:
                pred_dict[pred_node] = no_filter_columns
                no_filter_columns += 1

        if not train_udf_graph_against_udf_runtime:
            adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, RET_to_filter_edges,
                                  logical_preds,
                                  plan_dict, pred_dict,
                                  pred_node_type_id)

        if not train_udf_graph_against_udf_runtime:
            if len(column_to_output_column_edges) > 0:
                data_dict[('column', 'col_output_col', 'output_column')] = column_to_output_column_edges
            for u, v in output_column_to_plan_edges:
                v_node_id, d_v = plan_dict[v]
                data_dict[('output_column', 'to_plan', f'plan{d_v}')].append((u, v_node_id))
            for u, v in table_to_plan_edges:
                v_node_id, d_v = plan_dict[v]
                data_dict[('table', 'to_plan', f'plan{d_v}')].append((u, v_node_id))

        # also pass number of nodes per type
        max_depth, max_pred_depth = get_depths(plan_depths, predicate_depths)

        assert no_filter_columns == len(logical_preds) - sum(logical_preds)
        num_nodes_dict = {
            'column': len(column_features),
            'table': len(table_features),
            'output_column': len(output_column_features),
            'filter_column': no_filter_columns,
        }

        if not zs_paper_dataset:
            # Additions for UDFs
            udf_node_dict = {
                'INV': len(udf_node_features['INV']),
                'COMP': len(udf_node_features['COMP']),
                'RET': len(udf_node_features['RET']),
                'BRANCH': len(udf_node_features['BRANCH']),
                'LOOP': len(udf_node_features['LOOP']),
                'LOOPEND': len(udf_node_features['LOOPEND'])
            }
            num_nodes_dict = {**num_nodes_dict, **udf_node_dict}

        if not train_udf_graph_against_udf_runtime:
            num_nodes_dict = update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth,
                                                num_nodes_dict)

        if not zs_paper_dataset:
            # expand data_dict with udf specific nodes and columns
            for edge_key, edges in udf_internal_edges.items():
                ntype1, ntype2 = edge_key.split('_')
                data_dict[(ntype1, edge_key, ntype2)] = edges

            data_dict[('column', 'col_INV', 'INV')] = col_to_INVOC_edges
            data_dict[('column', 'col_COMP', 'COMP')] = col_to_COMP_edges
            data_dict[('RET', 'RET_outcol', 'output_column')] = RET_to_outcol_edges

            # assert ('RET', 'RET_filter', 'filter_column') not in data_dict.keys(), f'{data_dict.keys()}'
            # data_dict[('RET', 'RET_filter', 'filter_column')] = RET_to_filter_edges

        # check that all edges point to nodes which are existing
        for key, edges in data_dict.items():
            src_ntype = key[0]
            dst_ntype = key[2]

            for edge in edges:
                src_id = edge[0]
                dst_id = edge[1]

                assert src_id < num_nodes_dict[src_ntype], f'{src_id} >= {num_nodes_dict[src_ntype]} / {key}'
                assert dst_id < num_nodes_dict[dst_ntype], f'{dst_id} >= {num_nodes_dict[dst_ntype]} / {key}'

        # create graph
        try:
            graph: dgl.DGLHeteroGraph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        except Exception as e:
            print(num_nodes_dict, flush=True)
            # for key, val in data_dict.items():
            #     print(f'{key}: {len(val)}', flush=True)
            raise e
        graph.max_depth = max_depth
        graph.max_pred_depth = max_pred_depth

        # annotate the graph with out_degree information
        assert len(graph.ntypes) > 0, f'{graph.ntypes} / {data_dict.keys()}'
        for ntype in graph.ntypes:
            graph.nodes[ntype].data['out_degree'] = torch.zeros((graph.num_nodes(ntype=ntype),), dtype=torch.float32)

        assert len(graph.ndata['out_degree'].keys()) > 0, f'{graph.ndata["out_degree"].keys()}'

        for src_node, etype, dst_node in graph.canonical_etypes:
            deg = graph.out_degrees(etype=(src_node, etype, dst_node))
            assert graph.ndata['out_degree'][
                       src_node].shape == deg.shape, f'{graph.ndata["out_degree"][src_node]} != {deg.shape}'
            graph.ndata['out_degree'][src_node] += deg

        # initialize hidden state with 0s
        for ntype in graph.ntypes:
            graph.nodes[ntype].data['h'] = torch.zeros((graph.num_nodes(ntype=ntype), 128), dtype=torch.float32)

        features = collections.defaultdict(list)
        features.update(dict(column=column_features, table=table_features, output_column=output_column_features,
                             filter_column=[f for f, log_pred in zip(filter_features, logical_preds) if not log_pred],
                             ))

        if not plans_have_no_udf:
            features.update(
                dict(INV=udf_node_features['INV'], COMP=udf_node_features['COMP'], RET=udf_node_features['RET'],
                     BRANCH=udf_node_features['BRANCH'], LOOP=udf_node_features['LOOP'],
                     LOOPEND=udf_node_features['LOOPEND']))
        if not train_udf_graph_against_udf_runtime:
            # sort the plan features based on the depth
            for u, plan_feat in enumerate(plan_features):
                u_node_id, d_u = plan_dict[u]
                features[f'plan{d_u}'].append(plan_feat)

        # sort the predicate features based on the depth
        for pred_node_id, pred_feat in enumerate(filter_features):
            if not logical_preds[pred_node_id]:
                continue
            node_type, _ = pred_node_type_id(logical_preds, pred_dict, pred_node_id)
            features[node_type].append(pred_feat)

        features = postprocess_feats(features, num_nodes_dict)

        # rather deal with runtimes in secs
        labels_s = postprocess_labels(labels_ms)

        # add depth information to the udf sub-graph to allow topological mp
        if not zs_paper_dataset:
            graph = annotate_graph_with_depth_information(graph, ntypes_list=udf_node_types,
                                                          etypes_canonical_list=udf_canonical_edge_types,
                                                          max_depth=max_depth + 100)

        return graph, features, labels_s, stats, sample_idxs, udf_graph_bitmask
    except Exception as e:
        print(f'Error in postgres_plan_collator: {e}', flush=True)
        traceback.print_exc()
        raise e


def postprocess_labels(labels):
    labels = np.array(labels, dtype=np.float32)
    labels /= 1000
    # we do this later
    # labels = torch.from_numpy(labels)
    return labels


def postprocess_feats(features, num_nodes_dict):
    # convert to tensors, replace nan with 0
    for k in features.keys():
        v = features[k]
        v = np.array(v, dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0)
        v = torch.from_numpy(v)
        features[k] = v
    # filter out any node type with zero nodes
    features = {k: v for k, v in features.items() if k in num_nodes_dict}
    return features


def update_node_counts(max_depth, max_pred_depth, nodes_per_depth, nodes_per_pred_depth, num_nodes_dict):
    num_nodes_dict.update({f'plan{d}': nodes_per_depth[d] for d in range(max_depth + 1)})
    num_nodes_dict.update({f'logical_pred_{d}': nodes_per_pred_depth[d] for d in range(max_pred_depth + 1)})
    # filter out any node type with zero nodes
    # num_nodes_dict = {k: v for k, v in num_nodes_dict.items() if v > 0}
    return num_nodes_dict


def get_depths(plan_depths, predicate_depths):
    max_depth = max(plan_depths)
    max_pred_depth = 0
    if len(predicate_depths) > 0:
        max_pred_depth = max(predicate_depths)
    return max_depth, max_pred_depth


def adapt_predicate_edges(data_dict, filter_to_plan_edges, intra_predicate_edges, RET_to_filter_edges, logical_preds,
                          plan_dict, pred_dict,
                          pred_node_type_id_func):
    # convert to plan edges
    for u, v in filter_to_plan_edges:
        # transform plan node to right id and depth
        v_node_id, d_v = plan_dict[v]
        # transform predicate node to right node type and id
        node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)

        data_dict[(node_type, 'to_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    # convert intra predicate edges (e.g. column to AND)
    for u, v in intra_predicate_edges:
        u_node_type, u_node_id = pred_node_type_id_func(logical_preds, pred_dict, u)
        v_node_type, v_node_id = pred_node_type_id_func(logical_preds, pred_dict, v)
        data_dict[(u_node_type, 'intra_predicate', v_node_type)].append((u_node_id, v_node_id))

    for u, v in RET_to_filter_edges:
        v_node_type, v_node_id = pred_node_type_id_func(logical_preds, pred_dict, v)
        data_dict[('RET', 'RET_filter', v_node_type)].append((u, v_node_id))


def create_node_types_per_depth(plan_depths, plan_to_plan_edges):
    plan_dict = dict()
    nodes_per_depth = collections.defaultdict(int)
    for plan_node, d in enumerate(plan_depths):
        plan_dict[plan_node] = (nodes_per_depth[d], d)
        nodes_per_depth[d] += 1
    # create edge and node types depending on depth in the plan
    data_dict = collections.defaultdict(list)
    for u, v in plan_to_plan_edges:
        u_node_id, d_u = plan_dict[u]
        v_node_id, d_v = plan_dict[v]
        assert d_v < d_u
        data_dict[(f'plan{d_u}', f'intra_plan', f'plan{d_v}')].append((u_node_id, v_node_id))
    return data_dict, nodes_per_depth, plan_dict


def add_numerical_scalers(feature_statistics):
    for k, v in feature_statistics.items():
        if v.get('type') == str(FeatureType.numeric):
            scaler = RobustScaler()
            scaler.center_ = v['center']
            scaler.scale_ = v['scale']
            feature_statistics[k]['scaler'] = scaler


def pred_node_type_id(logical_preds, pred_dict, u):
    if logical_preds[u]:
        u_node_id, depth = pred_dict[u]
        node_type = f'logical_pred_{depth}'
    else:
        u_node_id = pred_dict[u]
        node_type = f'filter_column'
    return node_type, u_node_id
