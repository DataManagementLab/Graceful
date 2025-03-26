import argparse
import json
import os
import re
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import gast
import networkx as nx
import numpy as np
import pandas as pd
from python_graphs import control_flow
from python_graphs.control_flow import ControlFlowGraph, ControlFlowNode, BasicBlock
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.augment_deepdb_card import report_stats
from cross_db_benchmark.benchmark_tools.utils import load_schema_json
from deepdb.inference import DeepDBEstimator
from udf_graph.annotate_graph_info import enhanceUDFgraph_query, enhanceUDFgraph_card
from udf_graph.dbms_wrapper import DBMSWrapper
from udf_graph.helper import extract_from_Compare, extract_from_BinOp, extract_from_AssignExpr, extract_from_BoolOp, \
    get_lib_func
from udf_graph.intermed_graph_helper import getCFGpaths, create_int_graph, hash_block
from udf_graph.lib_embedding import get_string_vec

# Create dict that gives a mapping for the vector creation => so on which position a certain op is in the vector
OP_DICT_KEYS = [gast.gast.Add, gast.gast.Sub, gast.gast.Mult, gast.gast.MatMult, gast.gast.Div,
                gast.gast.Mod, gast.gast.Pow, gast.gast.LShift, gast.gast.RShift, gast.gast.BitOr, gast.gast.BitXor,
                gast.gast.BitAnd, gast.gast.FloorDiv]
COMP_OP_DICT_KEYS = [gast.gast.Eq, gast.gast.NotEq, gast.gast.Lt, gast.gast.LtE, gast.gast.Gt, gast.gast.GtE,
                     gast.gast.Is, gast.gast.IsNot, gast.gast.In, gast.gast.NotIn]
TYPE_DICT_KEYS = [float, int, str, bool]
OP_DICT = dict(zip(OP_DICT_KEYS,
                   ["Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow", "LShift", "RShift", "BitOr", "BitXor",
                    "BitAnd", "FloorDiv"]))
COMP_OP_DICT = dict(zip(COMP_OP_DICT_KEYS, ["Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn"]))
TYPE_DICT = dict(zip(TYPE_DICT_KEYS, np.arange(0, len(TYPE_DICT_KEYS))))
NO_OPS = len(OP_DICT.keys())
NO_COMP_OPS = len(OP_DICT.keys())
NO_TYPES = len(TYPE_DICT.keys())
SQL_DATA_TYPES = ["FLOAT", "INTEGER", "SMALLINT",
                  "DOUBLE PRECISION"]  # TODO: Rethink the representation here, e.g., INTEGER and INT are the same, we need to maps this later on
NO_SQL_DATA_TYPES = len(SQL_DATA_TYPES)


def getUDFparams(UDF_node: ControlFlowNode, nx_graph: nx.DiGraph) -> Tuple[Dict[str, int], nx.DiGraph]:
    """
    Extracts the input parameters of a UDF and adds them to the nx_graph
    :param UDF_node: first node in the graph (assumed to be FunctionDef node)
    :param nx_graph: graph to which VAR nodes should be added
    :return: dictionary mapping variable_name to node_id and updated nx_graph
    """
    assert isinstance(UDF_node.instruction.node,
                      gast.gast.FunctionDef), f"UDF_node is not of type FunctionDef: {type(UDF_node.instruction.node)} / {UDF_node.instruction.node}"

    var_name_node_id_dict = {}
    for index, arg in enumerate(UDF_node.instruction.node.args.args):
        assert isinstance(arg, gast.gast.Name), f"UDF_node is not of type Name: {type(arg)} / {arg}"

        input_var_name: str = arg.id

        var_name_node_id_dict[input_var_name] = index
        nx_graph.add_node(index, type="VAR", var_id=input_var_name)

    return var_name_node_id_dict, nx_graph


def getUDFgraph(CFgraph: ControlFlowGraph, source_code: str, is_duckdb: bool,
                add_loop_end_node: bool = False) -> nx.DiGraph:
    # obtain information about the variables/parameters of the UDF
    var_name_node_id_dict, dir_graph = getUDFparams(CFgraph.nodes[0], nx.DiGraph())

    # call helper functions to obtain the intermediate graph (multiple paths from start to end, each path is a list of blocks, loops have been removed)
    CFG_paths: List[List[BasicBlock]] = getCFGpaths(CFgraph)

    # create an intermediate graph which contructs a networkx graph on the block level
    intermed_graph, node2block, block2node = create_int_graph(CFG_paths)

    # set up some variables that are needed later
    to_visit = []  # stores the node ids that have to be visited next
    start_lookup = {}  # key: node from intermed_graph, value: start position
    visited_nodes = []
    alias2lib_map = {}  # e.g., needed for: import numpy as np => np -> numpy
    func2lib_map = {}  # needed for: from numpy import max => max -> numpy
    libnames = []  # keep track of all libraries that are imported

    if is_duckdb:
        libnames = ['numpy', 'math']

    colormap = []
    loop_present = False
    nested_loop = False
    counter = len(var_name_node_id_dict.keys())

    # initialize to_visit and start_lookup with information from the starting node 0
    to_visit.append(0)  # node 0 is the starting node, hence add it to to_visit
    start_lookup[0] = [counter]  # set the starting position to the node following all variable nodes

    loop_head_ids = []

    # dictionary storing number of unprocessed incoming edges of blocks in intermed graph
    # key: block id, value: number of unprocessed incoming edges
    unprocessed_incoming_edges = defaultdict(int)
    for node in intermed_graph.nodes:
        unprocessed_incoming_edges[node] = len(intermed_graph.in_edges(node))

    while len(to_visit) > 0:
        # get the next node from the intermed graph that has to be visited
        current_node = None
        for ptr in range(len(to_visit)):
            if unprocessed_incoming_edges[to_visit[ptr]] == 0:
                current_node = to_visit.pop(ptr)
                break

        if current_node is None:
            raise Exception("This should not happen. No node found which has no unprocessed incoming edges")

        # lookup the code-block that belongs to the current node of the intermediate graph
        current_block = node2block[current_node]
        prev_node = start_lookup[current_node]

        # in case no loop-end should be constructed, carry over prev nodes
        loop_end_prev_carryover = None

        if current_block.label == "test_block":
            assert not nested_loop, "a third while loop is not supported in the featurization"
            if loop_present:
                # we are in a nested loop
                nested_loop = True
            else:
                # we are in a loop
                loop_present = True
        elif current_block.label == "<loop_end>":
            assert loop_present, "loop_end without loop_start"

            if len(loop_head_ids) > 1:
                assert nested_loop, "loop_end without nested loop"

            loop_head_id = loop_head_ids.pop(-1)  # remove the last element from the list

            if nested_loop:
                nested_loop = False
            elif loop_present:
                loop_present = False

            if add_loop_end_node:
                loop_head_node = dir_graph.nodes[loop_head_id]

                # add a node that represents the end of a loop
                dir_graph.add_node(counter, block_hash=hash_block(current_block),
                                   fixed_iter=loop_head_node['fixed_iter'], no_iter=loop_head_node['no_iter'],
                                   loop_type=loop_head_node['loop_type'],
                                   loop_part=str(loop_present), type="LOOP_END")
                if prev_node is not None:
                    while len(prev_node) > 0:
                        dir_graph.add_edge(prev_node.pop(0), counter)
                dir_graph.add_edge(loop_head_id, counter)
                prev_node.append(counter)
                counter += 1
            else:
                loop_end_prev_carryover = prev_node

        # iterate over all nodes that belong to the current block
        for index, cf_node in enumerate(current_block.control_flow_nodes):
            if current_block.label == "<loop_end>":
                # skip all cf nodes in loop_end, since they are only for generating a unique hash value
                continue

            if cf_node in visited_nodes:  # make sure to not visit a certain node twice
                continue
            visited_nodes.append(cf_node)  # append current node to the list of visited nodes

            # Handling of nodes that deal with importing of library functions
            if type(cf_node.instruction.node) == gast.gast.Import:
                for elem in cf_node.instruction.node.names:
                    libnames.append(elem.name)
                    # if import like this: import numpy => asname = None
                    # if import like this: import numpy as np => asname = 'np'
                    if elem.asname is not None:
                        alias2lib_map[elem.asname] = elem.name
            elif type(cf_node.instruction.node) == gast.gast.ImportFrom:
                libname = cf_node.instruction.node.module
                libnames.append(libname)
                for elem in cf_node.instruction.node.names:
                    func_name = elem.name
                    func2lib_map[func_name] = libname

            # if the last node of block is a compare instruction, then we know that we are at a branch node
            # also, if there are if statement with "and", "or" etc., then we will see a BoolOP
            elif current_block.label != "test_block" and (index == len(current_block.control_flow_nodes) - 1 and (
                    type(cf_node.instruction.node) == gast.gast.Compare or type(
                cf_node.instruction.node) == gast.gast.BoolOp)):
                compare_ops = np.zeros(NO_COMP_OPS)  # embedding vector for the compare operation(s)
                comp_datatypes = np.zeros(
                    NO_TYPES)  # embedding vector for the datatypes involved in the compare operation(s)
                cmops = []
                dtypes = []
                varnames = []  # TODO: Varnames are extracted, decision has to be made on how to use it later for card. est.
                bool_connectors = []  # TODO: Connectors are extracted, decision has to be made on how to use it later for card. est.

                if type(cf_node.instruction.node) == gast.gast.Compare:
                    cmop, dtype, varname = extract_from_Compare(cf_node.instruction.node)
                    cmops.append(cmop)
                    dtypes.append(dtype)
                    varnames += varname
                elif type(cf_node.instruction.node) == gast.gast.BoolOp:  # BoolOp
                    bool_connectors, cmops, dtypes, varnames = extract_from_BoolOp(cf_node.instruction.node)
                else:
                    raise Exception("This should not happen")

                # for each elem in ops, retrieve the key from OP_DICT
                cmops = [COMP_OP_DICT[elem] for elem in cmops]

                # in if, we only have one compare operation hence we can take the only element from the list
                dir_graph.add_node(counter, block_hash=hash_block(current_block),
                                   lineno=cf_node.instruction.node.lineno,
                                   cmops=cmops[0], loop_part=str(False), type="BRANCH")

                if prev_node is not None:
                    while len(prev_node) > 0:
                        dir_graph.add_edge(prev_node.pop(0), counter)
                prev_node.append(counter)
                counter += 1
                if not loop_present:
                    colormap.append("yellow")  # use color mapping similar to PowerPoint slides
                else:
                    # indicate that the last inserted node is part of a loop
                    nx.set_node_attributes(dir_graph, {counter - 1: str(True)}, name="loop_part")
                    colormap.append("orange")

            # set a final return node when a return statement occurs at the end of a code block
            elif index == len(current_block.control_flow_nodes) - 1 and type(
                    cf_node.instruction.node) == gast.gast.Return:
                subtree = cf_node.instruction.node
                libs, ops, lib_args, op_args = [], [], [], []
                if hasattr(subtree, "value") and type(subtree.value) == gast.gast.Call:
                    libs, lib_args = get_lib_func(subtree.value, alias2lib_map, func2lib_map, libnames)
                elif hasattr(subtree, "value") and type(subtree.value) == gast.gast.BinOp:
                    ops, dts, libs, lib_args, op_args = extract_from_BinOp(cf_node.instruction.node.value,
                                                                           alias2lib_map,
                                                                           func2lib_map, libnames)

                # if there are calls or
                for idx, lib in enumerate(libs):
                    lib_vec = get_string_vec(lib)
                    # ops is zero in this case since this node only captures the library call
                    # to combination of several library calls (e.g., math.sqrt() + math.log() is performed
                    # at the node following the library call nodes
                    dir_graph.add_node(counter, block_hash=hash_block(current_block),
                                       lineno=cf_node.instruction.node.lineno, lib_embedding=lib_vec, lib_onehot=lib,
                                       ops=[],
                                       loop_part=str(False),
                                       type="COMP")

                    # add edges from the variable to the respective function where this function is involved
                    for arg_lst in lib_args[idx]:
                        for arg in arg_lst:
                            if arg in var_name_node_id_dict.keys():  # check if it is a input variable
                                dir_graph.add_edge(var_name_node_id_dict[arg], counter)

                    colormap.append("red")
                    if prev_node is not None:
                        while len(prev_node) > 0:
                            dir_graph.add_edge(prev_node.pop(0), counter)
                    prev_node.append(counter)
                    counter += 1

                # check if there are normal ops present in the return stmt; if so, then add a new node for covering the ops
                if len(ops) > 0:

                    # for each elem in ops, retrieve the key from OP_DICT
                    ops = [OP_DICT[elem] for elem in ops]

                    # libs is set to a vector of zeros with 3072 entries since the normal embedding vector has 3072 entries
                    dir_graph.add_node(counter, block_hash=hash_block(current_block),
                                       lineno=cf_node.instruction.node.lineno, lib_embedding=np.zeros(3072),
                                       lib_onehot='null',
                                       ops=ops,
                                       loop_part=str(False), type="COMP")

                    # append the necessary edges from col to the operator node
                    for arg in op_args:
                        if arg in var_name_node_id_dict.keys():  # check if it is a input variable
                            dir_graph.add_edge(var_name_node_id_dict[arg], counter)

                    colormap.append("red")
                    if prev_node is not None:
                        while len(prev_node) > 0:
                            dir_graph.add_edge(prev_node.pop(0), counter)
                    prev_node.append(counter)
                    counter += 1

                dir_graph.add_node(counter, block_hash=hash_block(current_block), type="RETURN")
                if prev_node is not None:
                    # dir_graph.add_edge(prev_node, counter)
                    while len(prev_node) > 0:
                        node_id = prev_node.pop(0)
                        assert dir_graph.nodes[node_id][
                                   "type"] != "INVOCATION", f"There should not be an edge between invocation node and return node {index}/{len(current_block.control_flow_nodes)}"
                        assert dir_graph.nodes[node_id][
                                   "type"] != "LOOP_HEAD", f"There should not be an edge between loop head and return node {index}/{len(current_block.control_flow_nodes)}"
                        dir_graph.add_edge(node_id, counter)
                colormap.append("black")
                counter += 1
            else:
                # the first node in a python_graphs CFG is always a node describing the arguments
                # we can use this node as our invocation node for our graph
                if type(cf_node.instruction.node) == gast.gast.arguments:
                    dir_graph.add_node(counter, block_hash=hash_block(current_block), type="INVOCATION")

                    # establish the connection between the nodes for the variables and the invocation node
                    for var in var_name_node_id_dict.keys():
                        dir_graph.add_edge(var_name_node_id_dict[var], counter)

                    prev_node.append(counter)
                    counter += 1
                    colormap.append("gray")  # use color mapping similar to PowerPoint slides
                else:
                    if cf_node.block.label == "test_block":  # handle loop header nodes
                        lineno = cf_node.instruction.node.lineno
                        source_at_line = source_code.split("\n")[lineno - 1]

                        if "for" in source_at_line:
                            loop_type = "for"
                        elif "while" in source_at_line:
                            loop_type = "while"
                        else:
                            raise Exception(f"Unknown loop type: {source_at_line}")

                        # determination of number of iteration
                        extracted_numbers = [int(s) for s in re.findall(r'\b\d+\b', source_at_line)]
                        if extracted_numbers != []:
                            no_iter = extracted_numbers[-1]
                            fixed_iter = True
                        else:
                            no_iter = -1  # maybe think if we need to find a different representation for the case that we cannot extract a number
                            fixed_iter = False

                        dir_graph.add_node(counter, block_hash=hash_block(current_block),
                                           lineno=cf_node.instruction.node.lineno, ops=ops, loop_part=str(nested_loop),
                                           fixed_iter=str(fixed_iter), no_iter=no_iter, loop_type=loop_type,
                                           type="LOOP_HEAD")
                        loop_head_ids.append(counter)
                        if prev_node is not None:
                            while len(prev_node) > 0:
                                dir_graph.add_edge(prev_node.pop(0), counter)
                        prev_node.append(counter)
                        counter += 1

                    else:
                        ops, dts, libs, lib_args, op_args = [], [], [], [], []
                        op_vec = np.zeros(NO_OPS)  # create a vector that has NO_OPS elems with all 0s
                        lib_vec = np.zeros(3072)
                        # Check type of node value and decide which helper function should be called
                        if hasattr(cf_node.instruction.node, 'value') and type(
                                cf_node.instruction.node.value) == gast.gast.BinOp:
                            ops, dts, libs, lib_args, op_args = extract_from_BinOp(cf_node.instruction.node.value,
                                                                                   alias2lib_map, func2lib_map,
                                                                                   libnames)
                        elif type(cf_node.instruction.node) == gast.gast.Assign or type(
                                cf_node.instruction.node) == gast.gast.Expr:
                            ops, dts, libs, lib_args = extract_from_AssignExpr(cf_node.instruction.node, alias2lib_map,
                                                                               func2lib_map, libnames)
                        elif type(cf_node.instruction.node) == gast.gast.AugAssign:
                            ops.append(gast.gast.Add)

                        if len(libs) > 1:
                            for idx, lib in enumerate(libs):  # create additional nodes if there are >1 library calls
                                lib_vec = get_string_vec(lib)
                                # ops is zero in this case since this node only captures the library call
                                # to combination of several library calls (e.g., math.sqrt() + math.log() is performed
                                # at the node following the library call nodes
                                dir_graph.add_node(counter, block_hash=hash_block(current_block),
                                                   lineno=cf_node.instruction.node.lineno, lib_embedding=lib_vec,
                                                   lib_onehot=lib,
                                                   ops=[],
                                                   loop_part=str(False), type="COMP")

                                # add edges from the variable to the respective function where this function is involved
                                for arg in lib_args[idx]:
                                    if arg in var_name_node_id_dict.keys():  # check if it is a input variable
                                        dir_graph.add_edge(var_name_node_id_dict[arg], counter)

                                if prev_node is not None:
                                    while len(prev_node) > 0:
                                        dir_graph.add_edge(prev_node.pop(0), counter)
                                prev_node.append(counter)
                                counter += 1

                                if not loop_present:
                                    colormap.append("red")  # use color mapping similar to PowerPoint slides
                                else:
                                    # indicate that the last inserted node is part of a loop
                                    nx.set_node_attributes(dir_graph, {counter - 1: str(True)}, name="loop_part")
                                    colormap.append("orange")

                            # we only need this node if the libs are connected via standard ops; otherwise we do not need it
                            if len(ops) > 0:
                                # lib can be none in this case since all lib call are incorporated in the prev. nodes
                                # add a np.zeros with 3072 entries since this is the normal lenght of the lib embedding
                                # for each elem in ops, retrieve the key from OP_DICT
                                ops = [OP_DICT[elem] for elem in ops]
                                dir_graph.add_node(counter, block_hash=hash_block(current_block),
                                                   lineno=cf_node.instruction.node.lineno, lib_embedding=np.zeros(3072),
                                                   lib_onehot='null', ops=ops,
                                                   loop_part=str(False), type="COMP")

                                # append the necessary edges from col to the operator node
                                for arg in op_args:
                                    if arg in var_name_node_id_dict.keys():  # check if it is a input variable
                                        dir_graph.add_edge(var_name_node_id_dict[arg], counter)

                                if prev_node is not None:
                                    while len(prev_node) > 0:
                                        dir_graph.add_edge(prev_node.pop(0), counter)
                                prev_node.append(counter)
                                counter += 1
                                if not loop_present:
                                    colormap.append("red")  # use color mapping similar to PowerPoint slides
                                else:
                                    # indicate that the last inserted node is part of a loop
                                    nx.set_node_attributes(dir_graph, {counter - 1: str(True)}, name="loop_part")
                                    colormap.append("orange")
                        else:
                            if len(libs) > 0:
                                # get_string_vec computes the feature vector for a library call, e.g., "math.sqrt"
                                lib_vec = get_string_vec(libs[0])

                            # for each elem in ops, retrieve the key from OP_DICT
                            ops = [OP_DICT[elem] for elem in ops]
                            dir_graph.add_node(counter, block_hash=hash_block(current_block),
                                               lineno=cf_node.instruction.node.lineno, lib_embedding=lib_vec,
                                               lib_onehot=libs[0] if len(libs) > 0 else 'null',
                                               ops=ops, loop_part=str(False), type="COMP")

                            # append the necessary edges from col to the operator node
                            for arg in op_args:
                                if arg in var_name_node_id_dict.keys():  # check if it is a input variable
                                    dir_graph.add_edge(var_name_node_id_dict[arg], counter)

                            if lib_args != []:
                                # add edges from the variable to the respective function where this function is involved
                                for arg in lib_args[0]:
                                    if arg in var_name_node_id_dict.keys():  # check if it is a input variable
                                        dir_graph.add_edge(var_name_node_id_dict[arg], counter)

                            if prev_node is not None:
                                while len(prev_node) > 0:
                                    dir_graph.add_edge(prev_node.pop(0), counter)
                            prev_node.append(counter)
                            counter += 1
                            if not loop_present:
                                colormap.append("red")  # use color mapping similar to PowerPoint slides
                            else:
                                # indicate that the last inserted node is part of a loop
                                nx.set_node_attributes(dir_graph, {counter - 1: str(True)}, name="loop_part")
                                colormap.append("orange")

        successors = list(intermed_graph.successors(current_node))
        # sort the list of successors based on the block_label (to make sure that order is always the same)
        successors = sorted(successors, key=lambda x: intermed_graph.nodes[x][
                                                          "block_label"] == 'true_block')  # true block should always be the first if present

        for child in successors:
            # decrement the number of unprocessed incoming edges of the child node
            unprocessed_incoming_edges[child] -= 1

            to_visit_tmp = []
            if child not in to_visit:
                to_visit_tmp.append(child)

            # add nodes to start of the list to make sure that these paths here are processed first to the end
            to_visit = to_visit_tmp + to_visit

            # update the start_lookup dictionary
            if child not in start_lookup.keys():
                start_lookup[child] = []
            start_lookup[child].append(counter - 1)

            if loop_end_prev_carryover is not None:
                assert len(successors) == 1, "There should be only one successor in case of loop_end"
                start_lookup[child].extend(loop_end_prev_carryover)

    def remove_unecssary_COMP_nodes(dir_graph):

        def get_succ_pred(node, dir_graph):
            ret_succ, ret_pred = [], []
            for succ in dir_graph.successors(node):
                ret_succ.append(succ)
            for pred in dir_graph.predecessors(node):
                ret_pred.append(pred)
            return ret_succ, ret_pred

        remove_nodes = []
        for node in dir_graph.nodes._nodes.keys():
            if dir_graph.nodes[node]["type"] == "COMP":
                if len(dir_graph.nodes[node]["ops"]) == 0 and np.count_nonzero(
                        dir_graph.nodes[node]["lib_embedding"]) == 0:
                    succs, preds = get_succ_pred(node, dir_graph)
                    # print(f"Remove node {node} since it is not necessary {dir_graph.nodes[node]}")
                    for succ in succs:
                        for pred in preds:
                            dir_graph.add_edge(pred, succ)

                    if "branch_cond" in dir_graph.nodes[node].keys():
                        nx.set_node_attributes(dir_graph, {succ: dir_graph.nodes[node]["branch_cond"]},
                                               name="branch_cond")

                    remove_nodes.append(node)

        for node in remove_nodes:
            dir_graph.remove_node(node)

        return dir_graph

    def retrieve_branch_cond(string: str) -> str:
        """
        Helper function that transform a Python if statement into SQL condition

        Example on how an if condition looks like:
            if occupation == 100:
        """

        assert string.strip().startswith("if"), f"String does not start with if: {string}"

        # first remove if keyword and the colon at the end
        ret_string = string.lstrip()[3:-1]
        if "==" in ret_string:  # sql uses only one equal sign
            ret_string = ret_string.replace("==", "=")

        # keep in mind: currently, we only have if statements on numerical columns
        # surround all words that start with a letter with "" => occupation => "occupation"
        # This is necessary to handle the cases in the SQL query later on
        ret_string = re.sub(r"\b([a-zA-Z]\w*)", r'"\1"', ret_string)
        return ret_string

    def attach_branch_conditions(dir_graph, intermed_graph, source_code, no_input_vars):
        """
        Helper function that attaches the SQL branching condition to the graph nodes that follow an if statement
        """

        def get_pred(node, graph):
            """
            Helper function that returns the first predecessors
            We know that there is only one predecessor, since we can just pick the first element
            """
            out_list = []
            for pred in graph.predecessors(node):
                if graph.nodes[pred]["type"] == "VAR":
                    continue
                else:
                    out_list.append(pred)
            return out_list[0]
            # return list(graph.predecessors(node))[0]

        visited_blocks = []

        for node in dir_graph.nodes._nodes.keys():
            if dir_graph.nodes[node]["type"] != "VAR":
                block_hash = dir_graph.nodes[node]["block_hash"]

                if block_hash not in visited_blocks:
                    visited_blocks.append(block_hash)
                    int_graph_node = block2node[block_hash]
                    if intermed_graph.nodes[int_graph_node]["block_label"] in ["true_block", 'false_block']:
                        pred = dir_graph.nodes[get_pred(node, dir_graph)]
                        pred_lineno = pred['lineno']
                        if_code = source_code.split("\n")[pred_lineno - 1]
                        branch_cond = retrieve_branch_cond(if_code)  # negation of the condition for the false branch

                        if intermed_graph.nodes[int_graph_node]["block_label"] == "false_block":
                            branch_cond = "NOT " + branch_cond

                        nx.set_node_attributes(dir_graph, {node: branch_cond}, name="branch_cond")

        return dir_graph

    dir_graph = attach_branch_conditions(dir_graph, intermed_graph, source_code, len(var_name_node_id_dict.keys()))

    dir_graph = remove_unecssary_COMP_nodes(dir_graph)

    return dir_graph, colormap


def load_parsed_plan(*, func_name, table_name, code_dict: Dict[str, List[str]], graph_location, sql_query: str, db_name,
                     dbms_wrapper: DBMSWrapper, graph_kwargs: Dict, card_est_assume_lazy_eval: bool, pullup_udf: bool,
                     udf_intermed_pos: bool, query_plan: Dict, database_statistics: Dict,
                     deepdb_estimator: Optional[DeepDBEstimator], est_stats: List[Dict], duckdb_kwargs: Dict,
                     schema_relationships: Dict, skip_wj: bool = False, skip_deepdb: bool = False):
    """

    :param func_name: name of the udf
    :param table_name: the table the udf is operating on
    :param code_dict:
    :param graph_location:
    :param query_dict:
    :param db_name:
    :param dbms_wrapper:
    :param graph_kwargs:
    :param suffix:
    :return:
    """
    # print(f"Transforming {func_name} in {db_name}", flush=True)

    code: List[str] = code_dict[func_name]

    # IF WE DO NOT SURROUND THE UDF WITH TRY/CATCH
    start_pos, stop_pos = 0, 0

    dbms = dbms_wrapper.dbms
    if dbms == 'postgres':
        for index, line in enumerate(code):
            if "AS $$" in line:
                start_pos = index + 1
            if "return test_func" in line:
                stop_pos = index
                break
    elif dbms == 'duckdb':
        start_pos = 0
        for index, line in enumerate(code):
            if line.lstrip().startswith("return "):
                stop_pos = index + 1
                break

    # iterate over all lines that belong to the python code and append them to a string
    code_string = ""
    for line in code[start_pos:stop_pos]:
        code_string += line
        if code_string[-1] != "\n":
            code_string += "\n"

    # apply python graphs to python functions stored in "intermediate.py"
    # then run the graph creation procedure on the output of python graphs
    t0 = time.time()
    cf_graph: ControlFlowGraph = control_flow.get_control_flow_graph(code_string)
    t1 = time.time()
    try:
        udf_graph, cm = getUDFgraph(cf_graph, code_string, is_duckdb=dbms_wrapper.dbms == 'duckdb', **graph_kwargs)
    except Exception as e:
        print(f"Error while processing {func_name} in {db_name}", flush=True)
        raise e

    # count number of LOOP_HEAD nodes
    # loop_start_nodes = [node for node in udf_graph.nodes if udf_graph.nodes[node]["type"] == "LOOP_HEAD"]

    t2 = time.time()
    # give the created networkx into enhanceUDFgraph to add information about input/output datatypes
    udf_graph = enhanceUDFgraph_query(udf_graph, func_name, code[0], dbms=dbms)
    t3 = time.time()
    # enhance the graph with information about cardinalities
    udf_graph, contained_col_col_comparison = enhanceUDFgraph_card(udf_graph, table_name=table_name,
                                                                   dbms_wrapper=dbms_wrapper,
                                                                   sql_query=sql_query,
                                                                   schema_relationships=schema_relationships,
                                                                   card_est_assume_lazy_eval=card_est_assume_lazy_eval,
                                                                   pullup_udf=pullup_udf,
                                                                   udf_intermed_pos=udf_intermed_pos,
                                                                   query_plan=query_plan,
                                                                   database_statistics=database_statistics,
                                                                   deepdb_estimator=deepdb_estimator,
                                                                   est_stats=est_stats, duckdb_kwargs=duckdb_kwargs,
                                                                   skip_wj=skip_wj, verbose=False, skip_deepdb=skip_deepdb)
    t4 = time.time()

    suffix = []
    if 'add_loop_end_node' in graph_kwargs and graph_kwargs['add_loop_end_node']:
        suffix.append('loopend')
    if card_est_assume_lazy_eval:
        suffix.append('lazy')

    if len(suffix) > 0:
        suffix = '_'.join(suffix)
        suffix = f'.{suffix}'
    else:
        suffix = ''

    # finally, store the networkx graph in created_graphs and named it like "func_15.XXX"
    graph_file_location = os.path.join(graph_location, func_name + f"{suffix}.gpickle")
    nx.write_gpickle(udf_graph, graph_file_location)
    t5 = time.time()

    # print(
    #     f"Done with graph {graph_file_location} (total: {t5 - t0:.2f}s, cfg: {t1 - t0:.2f}s, getUDFgraph: {t2 - t1:.2f}, enhance: {t3 - t2:.2f}, enhanceCard: {t4 - t3:.2f}, write: {t5 - t4:.2f}",
    #     flush=True)

    timings = dict(total=t5 - t0, cfg=t1 - t0, getUDFgraph=t2 - t1, enhance=t3 - t2, enhanceCard=t4 - t3, write=t5 - t4)

    # Indicate that the graph was created successfully
    return graph_file_location, timings, contained_col_col_comparison


def prepareGraphs(*, code_location, graph_location, exp_folder, func_tab_map, db_name,
                  dbms_wrapper: DBMSWrapper, duckdb_kwargs: Dict, graph_kwargs: Dict, card_est_assume_lazy_eval: bool,
                  pullup_udf: bool, udf_intermed_pos: bool, skip_wj: bool = False, skip_deepdb: bool = False,
                  deepdb_rel_ensemble_location: str = None,
                  deepdb_single_ensemble_location: str = None):
    print(
        f"Preparing graphs for {db_name}: code_location={code_location}, graph_location={graph_location}, "
        f"exp_folder={exp_folder}, func_tab_map={func_tab_map}, db_name={db_name}")

    def create_query_dict(calls_path):
        out_dict = {}
        # open the calls.sql file that is stored at path and read it split by lines to an array
        with open(calls_path, "r") as f:
            calls = f.read().splitlines()
        UDF_regex = re.compile('func_((\d*)\([\w\W,\s]*?\))')

        for call in calls:
            # extract the udf name using the reqular expression
            UDF_match = UDF_regex.search(call)
            func_name = UDF_match.group(0).split("(")[0]

            # use the name as key and the query as value in the dictionary
            out_dict[func_name] = call

        return out_dict

    # check if we have to create the folder for the graphs first
    os.makedirs(graph_location, exist_ok=True)

    # read csv file from func_tab_map into data frame
    df = pd.read_csv(func_tab_map, sep=",")

    json_path = os.path.join(exp_folder, "parsed_plans", db_name, "workload.json")
    if not os.path.exists(json_path):
        print(f"Skipping dataset {db_name} since {json_path} does not exist")
        return

    # load the json file into a dictionary
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(os.path.join(code_location, 'udfs.json')) as f:
        code_dict = json.load(f)

    # Load parsed plans
    # TB: Loading does not work in parallel, so we have to do it sequentially. It seems that the loading is not compatible with process pools
    args = dict(code_dict=code_dict, graph_location=graph_location, db_name=db_name)

    global_timings_dict = defaultdict(int)
    global_ctr = 0

    # load deepdb instance
    if skip_deepdb:
        deepdb_estimator = None
    else:
        deepdb_estimator = DeepDBEstimator(
            ensemble_locations=[deepdb_single_ensemble_location, deepdb_rel_ensemble_location],
            db_name=db_name, scale=1)

    # load schema relationships
    schema = load_schema_json(dataset=db_name)
    schema_relationships = schema.relationships

    database_statistics = data['database_stats']
    est_stats = []

    udf_w_col_col_comparison = []
    for plan in tqdm(data["parsed_plans"]):
        func_name = plan["udf"]['udf_name']
        sql_query = plan['query']
        assert func_name is not None, f"func_name is None for {plan}"
        try:
            table_name = df.loc[df.iloc[:, 0] == func_name].iloc[:, 1].values[0]
            table_name = table_name.strip()
        except Exception as e:
            print(f'Func name: {func_name}', flush=True)
            print(df, flush=True)
            raise e
        graph_file_location, timings, contains_col_col_comparison = load_parsed_plan(**args, sql_query=sql_query,
                                                                                     func_name=func_name,
                                                                                     table_name=table_name,
                                                                                     dbms_wrapper=dbms_wrapper,
                                                                                     graph_kwargs=graph_kwargs,
                                                                                     card_est_assume_lazy_eval=card_est_assume_lazy_eval,
                                                                                     pullup_udf=pullup_udf,
                                                                                     udf_intermed_pos=udf_intermed_pos,
                                                                                     query_plan=plan,
                                                                                     deepdb_estimator=deepdb_estimator,
                                                                                     database_statistics=database_statistics,
                                                                                     est_stats=est_stats,
                                                                                     duckdb_kwargs=duckdb_kwargs,
                                                                                     schema_relationships=schema_relationships,
                                                                                     skip_wj=skip_wj,
                                                                                     skip_deepdb=skip_deepdb)
        if contains_col_col_comparison:
            udf_w_col_col_comparison.append(func_name)
        for key, value in timings.items():
            global_timings_dict[key] += value
        global_ctr += 1

    # dump udf names with column-column comparison to file
    with open(os.path.join(graph_location, 'udf_w_col_col_comparison.json'), 'w') as f:
        json.dump(udf_w_col_col_comparison, f)

    print(
        f"Plans: {global_ctr} | Total: {global_timings_dict['total']:.2f}s, cfg: {global_timings_dict['cfg']:.2f}s, getUDFgraph: "
        f"{global_timings_dict['getUDFgraph']:.2f}, enhance: {global_timings_dict['enhance']:.2f}, "
        f"enhanceCard: {global_timings_dict['enhanceCard']:.2f}, write: {global_timings_dict['write']:.2f}",
        flush=True)
    report_stats(est_stats)


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
