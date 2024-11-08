import dgl
import dgl.function as fn
import torch

from models.zero_shot_models.message_aggregators.aggregator import MessageAggregator


class MscnConv(MessageAggregator):
    """
    A message aggregator that sums up child messages and afterwards combines them with the current hidden state of the
    parent node using an MLP
    """

    def __init__(self, hidden_dim=0, **kwargs):
        super().__init__(input_dim=2 * hidden_dim, output_dim=hidden_dim, **kwargs)

    def forward(self, graph: dgl.DGLHeteroGraph = None, etypes=None, in_node_types=None, out_node_types=None,
                feat_dict=None):
        if len(etypes) == 0:
            return feat_dict

        with graph.local_scope():
            if self.test:
                # reduce features by degree - to avoid counting nodes twice
                # write to new dict to avoid overwriting the original dict. This would lead to repeadetly dividing by the degree if a node is used in multiple stages
                new_feat_dict = dict()
                for src_ntype, etype, dst_ntype in etypes:
                    degree = graph.ndata['out_degree'][src_ntype].reshape((-1, 1))
                    degree = torch.maximum(degree, torch.ones_like(degree))
                    new_feat_dict[src_ntype] = feat_dict[src_ntype] / degree

                # copy over features which have not changed
                for key, val in feat_dict.items():
                    if key not in new_feat_dict:
                        new_feat_dict[key] = val
                graph.ndata['h'] = new_feat_dict
            else:
                graph.ndata['h'] = feat_dict

            # message passing
            graph.multi_update_all({etype: (fn.copy_src('h', 'm'), fn.sum('m', 'ft')) for etype in etypes},
                                   cross_reducer='sum')

            feat = graph.ndata['h']
            rst = graph.ndata['ft']

            out_dict = self.combine(feat, out_node_types, rst)
            return out_dict
