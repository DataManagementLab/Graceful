import dgl
import torch
from torch import nn

from models.zero_shot_models.message_aggregators import message_aggregators
from models.zero_shot_models.topological_mp_layer import TopologicalMPLayer
from models.zero_shot_models.utils.fc_out_model import FcOutModel
from models.zero_shot_models.utils.node_type_encoder import NodeTypeEncoder


class ZeroShotModel(FcOutModel):
    """
    A zero-shot cost model that predicts query runtimes on unseen databases out-of-the-box without retraining.
    """

    def __init__(self, device='cpu', final_mlp_kwargs=None, output_dim=1,
                 tree_layer_kwargs=None, skip_message_passing=False, node_type_kwargs=None,
                 feature_statistics=None, add_tree_model_types=None, prepasses=None, post_udf_passes=None,
                 featurization=None,
                 encoders=None, label_norm=None, mp_ignore_udf: bool = False, return_graph_repr: bool = False,
                 return_udf_repr: bool = False, plans_have_no_udf: bool = False,
                 train_udf_graph_against_udf_runtime: bool = False, work_with_udf_repr: bool = False,
                 test_with_count_edges_msg_aggr: bool = False):

        super().__init__(output_dim=output_dim, final_out_layer=True, **final_mlp_kwargs)

        self.label_norm = label_norm

        # use different models per edge type
        self.skip_message_passing = skip_message_passing
        self.device = device

        self.mp_ignore_udf = mp_ignore_udf
        self.return_graph_repr = return_graph_repr
        self.return_udf_repr = return_udf_repr
        self.train_udf_graph_against_udf_runtime = train_udf_graph_against_udf_runtime
        self.work_with_udf_repr = work_with_udf_repr
        self.test_with_count_edges_msg_aggr = test_with_count_edges_msg_aggr

        # use different models per edge type
        if self.train_udf_graph_against_udf_runtime:
            tree_model_types = add_tree_model_types
        else:
            tree_model_types = add_tree_model_types + ['to_plan', 'intra_plan', 'intra_pred']
        tree_layer_name = tree_layer_kwargs.pop('tree_layer_name')

        mod_dict = {
            name: message_aggregators.__dict__[tree_layer_name](**tree_layer_kwargs,
                                                                test=self.test_with_count_edges_msg_aggr)
            for name in tree_model_types
        }
        self.msg_aggregators = nn.ModuleDict(mod_dict)
        self.plans_have_no_udf = plans_have_no_udf
        if plans_have_no_udf:
            self.topological_mp_layer = None
        else:
            self.topological_mp_layer = TopologicalMPLayer(tree_layer_kwargs=tree_layer_kwargs,
                                                           test_with_count_edges_msg_aggr=test_with_count_edges_msg_aggr)

        # these message passing steps are performed in the beginning (dependent on the concrete database system at hand)
        self.prepasses = prepasses
        self.post_udf_passes = post_udf_passes

        if featurization is not None:
            self.featurization = featurization
            # different models to encode plans, tables, columns, filter_columns and output_columns
            self.node_type_encoders = nn.ModuleDict({
                enc_name: NodeTypeEncoder(features, feature_statistics, **node_type_kwargs)
                for enc_name, features in encoders
            })

    def encode_node_types(self, g, features):
        """
        Initializes the hidden states based on the node type specific models.
        """
        # initialize hidden state per node type
        hidden_dict = dict()
        for node_type, input_features in features.items():
            # encode all plans with same model
            if node_type not in self.node_type_encoders.keys():
                assert node_type.startswith('plan') or node_type.startswith('logical_pred'), f'{node_type}'

                if node_type.startswith('logical_pred'):
                    node_type_m = self.node_type_encoders['logical_pred']
                else:
                    node_type_m = self.node_type_encoders['plan']
            else:
                node_type_m = self.node_type_encoders[node_type]
            try:
                if self.test_with_count_edges_msg_aggr:
                    hidden_dict[node_type] = torch.ones(input_features.shape[0], 1, device=self.device)
                elif len(input_features) == 0:
                    # no node of this type in graph
                    hidden_dict[node_type] = input_features
                elif len(input_features.shape) == 1:
                    raise Exception(f'Node type {node_type} has only one feature: {input_features}')
                    hidden_dict[node_type] = input_features  # todo: check if this is correct
                else:
                    hidden_dict[node_type] = node_type_m(input_features)
            except Exception as e:
                print(f'{node_type} / {input_features.shape}', flush=True)
                raise e

        return hidden_dict

    def forward(self, input):
        """
        Returns logits for output classes
        """
        graph, features = input

        # make sure there a the same number of RET and plan0 nodes in the batched graph
        if not self.plans_have_no_udf and graph.number_of_nodes('INV') > 0:
            assert graph.number_of_nodes('INV') == graph.number_of_nodes(
                'RET'), f'{graph.number_of_nodes("INV")} / {graph.number_of_nodes("RET")}'

            # we might train also on queries which have no udf
            # if not self.train_udf_graph_against_udf_runtime:
            #     assert graph.number_of_nodes('RET') == graph.number_of_nodes(
            #         'plan0'), f'{graph.number_of_nodes("RET")} / {graph.number_of_nodes("plan0")}'

        features = self.encode_node_types(graph, features)
        if self.return_udf_repr:
            graph_repr, udf_repr, feat_dict = self.message_passing(graph, features)
        else:
            graph_repr, feat_dict = self.message_passing(graph, features)
            udf_repr = None

        # feed them into final feed forward network
        if not self.test_with_count_edges_msg_aggr:
            out = self.fcout(graph_repr)
        else:
            out = graph_repr

        assert out.shape[0] == graph_repr.shape[0], f'{out.shape} / {graph_repr.shape}'
        if udf_repr is not None:
            assert out.shape[0] == udf_repr.shape[0], f'{out.shape} / {udf_repr.shape}'

        if self.return_graph_repr and self.return_udf_repr:
            return out, udf_repr, graph_repr, feat_dict
        elif self.return_graph_repr:
            return out, graph_repr, feat_dict
        elif self.return_udf_repr:
            return out, udf_repr, feat_dict
        else:
            return out, feat_dict

    def topological_mp(self, graph: dgl.DGLHeteroGraph, feat_dict, max_depth: int = 100):
        for depth in range(0, max_depth):
            # print(f'Performing topological message passing at depth {depth}', flush=True)
            feat_dict, edge_matching_depth_found = self.topological_mp_layer(graph, feat_dict, depth)
            # print(f'Edge matching depth found: {edge_matching_depth_found}', flush=True)
            # if no edge with the desired depth was found, we can stop the loop
            if not edge_matching_depth_found:
                break

        return feat_dict

    def message_passing(self, g, feat_dict):
        """
        Bottom-up message passing on the graph encoding of the queries in the batch. Returns the hidden states of the
        root nodes.
        """

        # also allow skipping this for testing
        if not self.skip_message_passing:
            # all passes before predicates, to plan and intra_plan passes
            pre_pass_directions = [
                PassDirection(g=g, **prepass_kwargs)
                for prepass_kwargs in self.prepasses
            ]

            post_udf_pass_directions = [
                PassDirection(g=g, **prepass_kwargs)
                for prepass_kwargs in self.post_udf_passes
            ]

            if not self.train_udf_graph_against_udf_runtime:
                if g.max_pred_depth is not None:
                    # intra_pred from deepest node to top node
                    for d in reversed(range(g.max_pred_depth)):
                        pd = PassDirection(model_name='intra_pred',
                                           g=g,
                                           e_name='intra_predicate',
                                           n_dest=f'logical_pred_{d}')
                        post_udf_pass_directions.append(pd)

                # filter_columns & output_columns to plan
                post_udf_pass_directions.append(PassDirection(model_name='to_plan', g=g, e_name='to_plan'))

                # intra_plan from deepest node to top node
                for d in reversed(range(g.max_depth)):
                    pd = PassDirection(model_name='intra_plan',
                                       g=g,
                                       e_name='intra_plan',
                                       n_dest=f'plan{d}')
                    post_udf_pass_directions.append(pd)

            def apply_mp_directions(pass_directions):
                # make sure all edge types are considered in the message passing
                combined_e_types = set()
                for pd in pass_directions:
                    combined_e_types.update(pd.etypes)
                # assert combined_e_types == set(g.canonical_etypes)
                for pd in pass_directions:
                    if len(pd.etypes) > 0:
                        try:
                            # check if there are nodes of the required types in the graph
                            in_match_found = False
                            out_match_found = False

                            for in_type in pd.in_types:
                                if in_type in feat_dict and feat_dict[in_type].shape[0] > 0:
                                    in_match_found = True
                                    break
                            for out_type in pd.out_types:
                                if out_type in feat_dict and feat_dict[out_type].shape[0] > 0:
                                    out_match_found = True
                                    break

                            # skip this pass if no nodes of the required types are in the graph
                            if not in_match_found or not out_match_found:
                                continue

                            out_dict = self.msg_aggregators[pd.model_name](g, etypes=pd.etypes,
                                                                           in_node_types=pd.in_types,
                                                                           out_node_types=pd.out_types,
                                                                           feat_dict=feat_dict)
                        except Exception as e:
                            print(f'Error in {pd.model_name} with {pd.etypes} / {pd.in_types} / {pd.out_types}')
                            for key, val in feat_dict.items():
                                print(f'{key} / {val.shape}')
                            raise e
                        for out_type, hidden_out in out_dict.items():
                            feat_dict[out_type] = hidden_out

            # all to udf passes
            apply_mp_directions(pre_pass_directions)
            # mp in udf
            if not self.mp_ignore_udf and not self.plans_have_no_udf:
                feat_dict = self.topological_mp(g, feat_dict)
            # mp in query plan
            apply_mp_directions(post_udf_pass_directions)

        # compute top nodes of dags
        if self.train_udf_graph_against_udf_runtime:
            out = feat_dict['RET']
        elif self.work_with_udf_repr:
            out = feat_dict['RET']
        else:
            out = feat_dict['plan0']

        if self.return_udf_repr:
            return out, feat_dict['RET'], feat_dict
        return out, feat_dict


class PassDirection:
    """
    Defines a message passing step on the encoded query graphs.
    """

    def __init__(self, model_name, g, e_name=None, n_dest=None, allow_empty=False):
        """
        Initializes a message passing step.
        :param model_name: which edge model should be used to combine the messages
        :param g: the graph on which the message passing should be performed
        :param e_name: edges are defined by triplets: (src_node_type, edge_type, dest_node_type). Only incorporate edges
            in the message passing step where edge_type=e_name
        :param n_dest: further restrict the edges that are incorporated in the message passing by the condition
            dest_node_type=n_dest
        :param allow_empty: allow that no edges in the graph qualify for this message passing step. Otherwise this will
            raise an error.
        """
        self.etypes = set()
        self.in_types = set()
        self.out_types = set()
        self.model_name = model_name

        for curr_n_src, curr_e_name, curr_n_dest in g.canonical_etypes:
            if e_name is not None and curr_e_name != e_name:
                continue

            if n_dest is not None and curr_n_dest != n_dest:
                continue

            self.etypes.add((curr_n_src, curr_e_name, curr_n_dest))
            self.in_types.add(curr_n_src)
            self.out_types.add(curr_n_dest)

        self.etypes = list(self.etypes)
        self.in_types = list(self.in_types)
        self.out_types = list(self.out_types)
        if not allow_empty:
            assert len(self.etypes) > 0, f"No nodes in the graph qualify for e_name={e_name}, n_dest={n_dest}"
