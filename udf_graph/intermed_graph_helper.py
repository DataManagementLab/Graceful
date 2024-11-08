from collections import defaultdict
from typing import List, Tuple, Dict, Iterator

import gast
import networkx as nx
from python_graphs.control_flow import BasicBlock, ControlFlowGraph


def hash_block(block: BasicBlock) -> int:
    uuid_lst = []
    # create a list of uuids of control flow nodes that belong to a block
    for node in block.control_flow_nodes:
        uuid_lst.append(node.uuid)
    return hash(block.label + str(uuid_lst))


def create_int_graph(CFGpaths: List[List[BasicBlock]]) -> Tuple[nx.DiGraph, Dict[int, BasicBlock], Dict[int, int]]:
    """
    Create an intermediate graph that represents the control flow of a function. This graph is used to create the final graph
    :param CFGpaths: list of paths
    :return: tuple of: intermediate graph, mapping from node_id to block, mapping from block_hash to node_id
    """
    dir_graph = nx.DiGraph()
    block2node = {}  # creates a mapping from the block hash to the node_id in the di_graph
    counter = 0
    node2block = {}

    for path in CFGpaths:  # iterate over all paths of our CFG
        prev_node = None
        for block in path:  # iterate over all blocks that belong to a path
            # get the hash of the current block
            block_hash = hash_block(block)
            if block_hash not in block2node.keys():  # add new node for a block
                block2node[block_hash] = counter

                # put a label on the true blocks so that we can identify later on
                if block.label == "true_block":  # = TRUE branch of an if statement
                    dir_graph.add_node(counter, block_label="true_block")
                elif block.label == "false_block":  # = TRUE branch of an if statement
                    dir_graph.add_node(counter, block_label="false_block")
                elif block.label == "test_block":
                    dir_graph.add_node(counter, block_label="loop_start")
                elif block.label == "<loop_end>":
                    dir_graph.add_node(counter, block_label="loop_end")
                else:
                    dir_graph.add_node(counter, block_label="NONE")

                node2block[counter] = block
                curr_node = counter
                counter += 1
            else:
                curr_node = block2node[block_hash]

            if prev_node is not None:
                dir_graph.add_edge(prev_node, curr_node)
            prev_node = curr_node

    def get_no_childs(child_iter: Iterator) -> int:
        """
        Count the number of elements in an iterator. Given an iterator over the childs of a node, the functions returns the number of children
        :param child_iter: iterator
        :return: count
        """
        counter = 0
        for child in child_iter:
            counter += 1
        return counter

    """
    Helper function that adds labels to the nodes that represent a false block given the knowledge about a true block
    """

    # def add_false_block(dir_graph: nx.DiGraph) -> nx.DiGraph:
    #     """
    #     Helper function that adds labels to the nodes that represent a false block given the knowledge about a true block
    #     :param dir_graph:
    #     :return:
    #     """
    #     for node in dir_graph.nodes._nodes.keys():
    #         # branch nodes have more than one sucessor
    #         if get_no_childs(dir_graph.successors(node)) > 1:
    #             for succ in dir_graph.successors(node):
    #                 if dir_graph.nodes[succ]["block_label"] == "NONE":
    #                     print(f'This should not happen: {node} / {succ.__dict__}',flush=True)
    #                     assert False, f'{node} / {succ.__dict__}'
    #                     nx.set_node_attributes(dir_graph, {succ: "false_block"}, name="block_label")
    #     return dir_graph
    #
    # dir_graph = add_false_block(dir_graph)
    return dir_graph, node2block, block2node


def get_no_eq_child(path: List[BasicBlock], comp_child: BasicBlock) -> int:
    """
    Search how often a given block occurs in a path. Block comparison is based on the UUIDs of the control flow nodes
    :param path: list of blocks
    :param comp_child: block under test
    :return: counts
    """

    # extract list of uuids of the control flow nodes that belong to the block
    comp_child_node_uuids = []
    for node in comp_child.control_flow_nodes:
        comp_child_node_uuids.append(node.uuid)
    comp_child_node_uuids = set(comp_child_node_uuids)

    uuid_paths = []
    for elem in path:
        helper = []
        for node in elem.control_flow_nodes:
            helper.append(node.uuid)
        uuid_paths.append(set(helper))

    # finally compare and return the number of equalities
    counter = 0
    for elem in uuid_paths:
        if elem == comp_child_node_uuids:
            counter += 1
    return counter


# returns a list of BasicBlocks that share the same path from the beginning to return
def getCFGpaths(CFgraph: ControlFlowGraph) -> List[List[BasicBlock]]:
    # get the entry nodes of the graph (are multiple)
    enter_node_generator = CFgraph.get_enter_control_flow_nodes()

    # check if the first node is a FunctionDef node
    n1 = next(enter_node_generator)
    assert isinstance(n1.instruction.node,
                      gast.gast.FunctionDef), f"UDF_node is not of type FunctionDef: {type(n1.instruction.node)} / {n1.instruction.node}"

    # skip first node and got to second
    n2 = next(enter_node_generator)

    # initialize the list of paths with the first block
    paths = [[n2.block]]
    out_paths = []

    while len(paths) > 0:
        new_paths = []
        # start at the first Block and append it to the list
        for path in paths:
            last_elem_in_path: BasicBlock = path[-1]

            # iterate over all children of the last block in the current path
            for child in last_elem_in_path.exits_from_end:
                assert isinstance(child, BasicBlock), f"child is not of type BasicBlock: {type(child)} / {child}"
                # new_path = copy.deepcopy(path)  # JW: this is not necessary, and super costly
                new_path = path[:]  # create new list that resembles the current path

                if len(new_path) > 0:
                    # make sure that the hashing of the last block is the same as the hashing of the last block of the current path
                    assert hash_block(new_path[-1]) == hash_block(path[-1])

                # check if the child is not more than once in the path: LOOP -> COMP -> LOOP -> RET would be fine
                no_eq_child = get_no_eq_child(new_path, child)
                if no_eq_child < 2:
                    new_path.append(child)
                    new_paths.append(new_path)  # append child to new path

        # check if one of the paths reached the end of the function (a return node)
        paths = []
        for path in new_paths:
            if path[-1].label == "<return>":
                # store this path safely
                out_paths.append(path)
            else:
                # put this path back into processing since it has not reached the end of the function yet
                paths.append(path)

    assert len(out_paths) > 0, "No paths found in the CFG"
    out_paths = check_paths(out_paths)
    out_paths = [x[:-1] for x in out_paths]  # remove the <return> node at the end
    out_paths = remove_loopback(out_paths)
    assert len(out_paths) > 0, "No paths found in the CFG"
    return out_paths


# simple helper function to check the validitiy of the paths
# we do not want to jump to the after block directly from the loop header
# Hence we want to see a that we have visited a body block before we jump from the last test_block to the rest of the func
def check_paths(out_paths: List[List[BasicBlock]]) -> List[List[BasicBlock]]:
    ret_paths = []
    for path in out_paths:
        test_blocks = []  # loop headers
        body_blocks = []  # loop childs

        for block in path:
            if block.label == "test_block":
                test_blocks.append(hash_block(block))
            if block.label == "body_block":
                body_blocks.append(hash_block(block))

        if len(test_blocks) > 0:
            # skip paths where we do not enter a loop and skip the body block
            if len(set(body_blocks)) == len(set(test_blocks)):
                ret_paths.append(path)
            else:
                # omit paths where we jump from the test block to the after block and skip the body block
                pass
        else:  # in case there are no test_blocks in the current path
            ret_paths.append(path)

    return ret_paths


def remove_loopback(paths: List[List[BasicBlock]]) -> List[List[BasicBlock]]:
    '''
    Remove the second occurrence of a test_block in a path (i.e. test -> body -> test -> ...) since this is a loop.
    And we will featurize only the first occurrence of the test_block
    :param paths:
    :return:
    '''
    for path in paths:
        helper_dict = defaultdict(int)

        # count how often a block occurs in a path
        for block in path:
            if block.label == "test_block":
                helper_dict[hash_block(block)] += 1

        if len(helper_dict) > 0:
            assert set(helper_dict.values()) == {2}, f"test_block have to occur twice in a path: {helper_dict.values()}"

        # remove all test_blocks that occur more than once (since this is a loop it will be visited multiple times in a path)
        for i in reversed(range(len(path))):
            if path[i].label == "test_block" and helper_dict[hash_block(path[i])] > 1:
                # here the order matters - first compute the hash of the block and then remove it from the path
                # otherwise the hash will be computed on a different block
                helper_dict[hash_block(path[i])] -= 1
                rm_block = path.pop(i)

                # mark the end of the loop with a special block (add stub control flow nodes to the block, to make sure that hash value is unique)
                loop_end_block = BasicBlock(label="<loop_end>")
                loop_end_block.control_flow_nodes = rm_block.control_flow_nodes
                path.insert(i, loop_end_block)

    return paths
