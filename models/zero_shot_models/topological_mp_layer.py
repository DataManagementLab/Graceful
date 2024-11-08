import dgl
import dgl.function as fn
import torch
from torch import nn

from models.zero_shot_models.message_aggregators import message_aggregators
from models.zero_shot_models.specific_models.udf_edge_types import udf_canonical_edge_types


class TopologicalMPLayer(torch.nn.Module):
    def __init__(self, tree_layer_kwargs, test_with_count_edges_msg_aggr: bool = False):
        super().__init__()
        new_topological_model_types = [etype for
                                       src_ntype, etype, dst_ntype
                                       in
                                       udf_canonical_edge_types]
        mod_dict = {
            name: message_aggregators.__dict__['MscnAggregator'](**tree_layer_kwargs,
                                                                 test=test_with_count_edges_msg_aggr)
            for name in new_topological_model_types
        }
        self.msg_aggregators = nn.ModuleDict(mod_dict)
        self.test = test_with_count_edges_msg_aggr

    def forward(self, graph: dgl.DGLHeteroGraph, orig_feat_dict, depth: int):
        # variable storing whether in this run over all etypes an edge with the desired depth was found
        edge_matching_depth_found = False

        # flag activating old implementation - the old implementation might has some bugs wherefor got replaced
        # by a more general implementation which comes with additional safeguards
        WORK_ON_GRAPH_FEATS: bool = False  # in general keep this deactivated

        PERFORM_MSG_AGGREGATION_IN_SEND_RECEIVE: bool = False

        if WORK_ON_GRAPH_FEATS:
            graph.ndata['h'] = orig_feat_dict
        else:
            feat_dict = orig_feat_dict.copy()
        with graph.local_scope():
            if WORK_ON_GRAPH_FEATS:
                out_dict = dict()
            else:
                out_dict = None

            # for src_node_type, etype, dst_node_type in graph.canonical_etypes:
            #     if src_node_type in udf_node_types and dst_node_type in udf_node_types:
            #         assert (src_node_type, etype, dst_node_type) in udf_canonical_edge_types, f'Edge type {etype} not in udf_canonical_edge_types: {udf_canonical_edge_types}'

            for src_node_type, etype, dst_node_type in udf_canonical_edge_types:
                # extract all edges with the desired depth
                e_ids = torch.nonzero(graph.edges[etype].data['depth'] == depth).reshape(-1)
                src_node_ids = graph.edges(etype=etype)[0][e_ids]
                dst_node_ids = graph.edges(etype=etype)[1][e_ids]

                # check if any edges with the desired depth were found for this etype
                if len(e_ids) == 0:
                    continue

                edge_matching_depth_found = True

                if not WORK_ON_GRAPH_FEATS:
                    src_node_ids, dst_node_ids = graph.find_edges(e_ids, etype=etype)
                    #
                    # assert torch.equal(src_node_ids0,src_node_ids)
                    # assert torch.equal(dst_node_ids0,dst_node_ids)

                    if self.test:
                        new_feat_dict = dict()

                        degree = graph.nodes[src_node_type].data['out_degree']
                        degree = torch.maximum(degree, torch.ones_like(degree))

                        # create bitmask from ids
                        cond = torch.zeros_like(degree, dtype=torch.bool)
                        cond[src_node_ids] = True

                        degree = degree.reshape((-1, 1))

                        degree_updated_sources = (feat_dict[src_node_type] / degree)[src_node_ids]
                        new_feat_dict[src_node_type] = feat_dict[src_node_type].clone()
                        new_feat_dict[src_node_type][src_node_ids] = degree_updated_sources

                        # copy over features which have not changed
                        for key, val in feat_dict.items():
                            if key not in new_feat_dict:
                                new_feat_dict[key] = val
                        graph.ndata['h'] = new_feat_dict
                    else:
                        assert not PERFORM_MSG_AGGREGATION_IN_SEND_RECEIVE
                        # overwrite graph states
                        graph.ndata['h'] = feat_dict
                        # graph.ndata['h'][src_node_type] = feat_dict[src_node_type]
                        # graph.ndata['h'][dst_node_type] = feat_dict[dst_node_type]

                def node_udf(nodes):
                    res = self.msg_aggregators[etype](
                        nodes.data['h'], nodes.data['ft'],
                    )
                    nodes.data['h'] = res
                    return nodes.data

                # message passing
                graph.send_and_recv(e_ids, fn.copy_src('h', 'm'), fn.sum('m', 'ft'),
                                    apply_node_func=node_udf if PERFORM_MSG_AGGREGATION_IN_SEND_RECEIVE or WORK_ON_GRAPH_FEATS else None,
                                    etype=etype)

                if WORK_ON_GRAPH_FEATS:
                    out_dict[dst_node_type] = graph.nodes[dst_node_type].data['h']
                else:
                    dst_nodes_to_update_bitmap = torch.zeros(graph.num_nodes(dst_node_type), dtype=torch.bool,
                                                             device=graph.device)
                    dst_nodes_to_update_bitmap[dst_node_ids] = True
                    dst_nodes_to_update_bitmap = dst_nodes_to_update_bitmap.reshape((-1, 1))

                    if PERFORM_MSG_AGGREGATION_IN_SEND_RECEIVE:
                        feat_dict[dst_node_type] = torch.where(dst_nodes_to_update_bitmap,
                                                               graph.nodes[dst_node_type].data['h'],
                                                               feat_dict[dst_node_type])
                    else:
                        feat = graph.nodes[dst_node_type].data['h']
                        # feat = feat_dict[dst_node_type]
                        rst = graph.nodes[dst_node_type].data['ft']

                        res = self.msg_aggregators[etype](feat, rst)

                        feat_dict[dst_node_type] = torch.where(dst_nodes_to_update_bitmap, res,
                                                               feat_dict[dst_node_type])
        if WORK_ON_GRAPH_FEATS:
            for key, value in orig_feat_dict.items():
                if key not in out_dict:
                    out_dict[key] = value
            return out_dict, edge_matching_depth_found
        else:
            return feat_dict, edge_matching_depth_found
