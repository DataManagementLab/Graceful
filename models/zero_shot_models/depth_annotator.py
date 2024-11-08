from typing import List, Tuple

import dgl
import torch


def annotate_graph_with_depth_information(graph: dgl.DGLGraph, ntypes_list: List[str],
                                          etypes_canonical_list: List[Tuple[str, str, str]], max_depth: int):
    """
    Annotates the graph with depth information. The depth of a node is the length of the longest path from the node to
    a leaf node. The depth of a leaf node is 0.
    """

    # get number of nodes per type for accurate offset information
    num_nodes_per_type = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes}
    num_edges_per_type = {etype: graph.number_of_edges(etype) for etype in
                          [etype for _, etype, _ in etypes_canonical_list]}

    # retrieve all source and target nodes for the involved edge types

    etype_srcs_dict = dict((etype, graph.edges(etype=etype)[0]) for _, etype, _ in etypes_canonical_list)
    etype_dsts_dict = dict((etype, graph.edges(etype=etype)[1]) for _, etype, _ in etypes_canonical_list)

    # count number of incoming edges for each node in graph to initialize the number of unprocessed incoming edges
    unprocessed_incoming_edges_per_ntype = dict(
        (ntype, torch.zeros(num_nodes_per_type[ntype])) for ntype in ntypes_list)

    # iterate over all edge types and sum up the number of incoming edges for each node
    for src_ntype, etype, dst_ntype in etypes_canonical_list:
        unprocessed_incoming_edges_per_ntype[dst_ntype] += torch.bincount(etype_dsts_dict[etype],
                                                                          minlength=num_nodes_per_type[dst_ntype])

    # bitmap storing which nodes have already been processed
    processed_nodes_per_ntype = dict((ntype, torch.zeros(num_nodes_per_type[ntype])) for ntype in ntypes_list)

    # depth information output tensor
    depth_per_ntype = dict((ntype, torch.zeros(num_nodes_per_type[ntype])) for ntype in ntypes_list)
    depth_per_etype = dict((etype, torch.zeros(num_edges_per_type[etype])) for _, etype, _ in etypes_canonical_list)

    # iterate over depth levels
    for i in range(max_depth):
        # retrieve all nodes that have no unprocessed incoming edges and still have not been processed
        iteration_nodes_to_process_per_ntype = dict()
        for ntype in ntypes_list:
            iteration_nodes_to_process_per_ntype[ntype] = torch.logical_and(
                (unprocessed_incoming_edges_per_ntype[ntype] == 0), (processed_nodes_per_ntype[ntype] == 0))

        # add node order information
        for ntype in ntypes_list:
            depth_per_ntype[ntype] += i * iteration_nodes_to_process_per_ntype[ntype]

        # update number of incoming edges for all nodes that are target of the processed nodes
        for src_ntype, etype, dst_ntype in etypes_canonical_list:
            edges_to_process = torch.gather(iteration_nodes_to_process_per_ntype[src_ntype], 0, etype_srcs_dict[etype])
            depth_per_etype[etype] += i * edges_to_process

            # select the src & dst nodes of the edges to process
            edge_sources = torch.masked_select(etype_srcs_dict[etype], edges_to_process)
            edge_destinations = torch.masked_select(etype_dsts_dict[etype], edges_to_process)

            # update the number of unprocessed incoming edges for the dst nodes
            processed_edges = torch.bincount(edge_destinations, minlength=num_nodes_per_type[dst_ntype])

            # update unprocessed edges counter
            unprocessed_incoming_edges_per_ntype[dst_ntype] -= processed_edges

        # mark nodes as processed
        for ntype in ntypes_list:
            processed_nodes_per_ntype[ntype] = torch.logical_or(processed_nodes_per_ntype[ntype],
                                                                iteration_nodes_to_process_per_ntype[ntype])

    # store depth information in graph
    for key, value in depth_per_etype.items():
        graph.edges[key].data['depth'] = value

    for key, value in depth_per_ntype.items():
        graph.nodes[key].data['depth'] = value

    return graph
