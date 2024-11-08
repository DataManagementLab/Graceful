import sys
from typing import List, Dict, Any

sys.path.append("../..")
import random
import numpy as np
from treelib import Tree
import string

OPERATORS = ["*", "/", "+", "-"]
COMP_OPS = ["==", "<=", "<", ">", ">=", "!="]


def numeric_statement(variables, stmt_operands, libs, lib_func_df_dict: Dict[str, Any]):
    imports = []

    use_lib = int(np.random.randint(0, 2, 1))  # decide whether to use a lib func or not
    if use_lib == 0:
        scalar_vals = []
        if stmt_operands - len(variables) > 0:
            scalar_vals = list(np.random.randint(1, 100, stmt_operands - len(
                variables)))  # watch out that stmt_operands is > len(variables)
            scalar_vals = [str(x) for x in scalar_vals]
        operands_lst = scalar_vals + variables
        random.shuffle(operands_lst)
        stmt = operands_lst.pop()
        for elem in operands_lst:
            random_pos = int(np.random.randint(0, len(OPERATORS), 1))
            stmt += str(OPERATORS[random_pos])
            stmt += str(elem)
    else:
        stmt = ""
        lib = libs[int(np.random.randint(0, len(libs), 1))]
        df = lib_func_df_dict[lib]
        no_ops = int(np.random.randint(1, 4, 1))

        for i in range(0, no_ops):
            random_func_idx = int(np.random.randint(0, len(df), 1))

            func_entry = df.iloc[random_func_idx]
            output = func_entry["Output"]

            while output != "Number" and no_ops > 1:  # Make sure that we have only numbers when combining multiple calls
                random_func_idx = int(np.random.randint(0, len(df), 1))
                func_entry = df.iloc[random_func_idx]
                output = func_entry["Output"]

            func_name = func_entry["Name"]
            no_args = func_entry["No_Args"]

            if no_args == "Many":
                if "Integer" in func_entry["Args"]:
                    args = np.random.randint(1, 100, 5)
                else:
                    args = np.random.uniform(1, 100, 5)
            # exact amount of args known
            else:
                if "Integer" in func_entry["Args"]:
                    args = np.random.randint(1, 100, int(no_args))
                else:
                    if len(variables) > 0:
                        if len(variables) >= int(no_args):
                            args = list(np.random.choice(variables, int(no_args)))
                        else:
                            args = variables
                            args += list(np.random.uniform(1, 100, int(no_args) - len(variables)))
                    else:
                        args = np.random.uniform(1, 100, int(no_args))
            arg_str = ""
            for arg in args:
                arg_str += str(arg) + ", "
            arg_str = arg_str[:-2]

            stmt += lib + "." + func_name + "(" + arg_str + ")"

            stmt += random.choice(OPERATORS)
            imports.append("import " + lib)

        stmt = stmt[:-1]  # remove the last char since this operator is not needed

    return stmt, imports


def string_statement(variables, string_funcs_df, max_len=100):
    # generate random string
    rand_str = generate_random_string(max_len)
    random_func_idx = int(np.random.randint(0, len(string_funcs_df), 1))
    func_name = string_funcs_df.iloc[random_func_idx]["Name"]
    args = string_funcs_df.iloc[random_func_idx]["Args"]

    if len(variables) > 0:
        rand_str_var = insert_var_to_rand_str(rand_str, variables[int(np.random.randint(0, len(variables), 1))])
        if args == "String":
            # pick a random char from str to serve as argument in the call
            random_char_arg = rand_str[int(np.random.randint(0, len(rand_str), 1))]
            if func_name != "replace":
                stmt = rand_str_var + "." + func_name + "(" + "\"" + random_char_arg + "\"" + ")"
            else:
                stmt = rand_str_var + "." + func_name + "(" + "\"" + random_char_arg + "\"" + ", \"random\"" + ")"
        elif args == "Integer":
            int_arg = len(rand_str) * 2
            stmt = rand_str_var + "." + func_name + "(" + str(int_arg) + ")"
        else:
            stmt = rand_str_var + "." + func_name + "()"
    else:
        if args == "String":
            # pick a random char from str to serve as argument in the call
            random_char_arg = rand_str[int(np.random.randint(0, len(rand_str), 1))]
            if func_name != "replace":
                stmt = "\"" + rand_str + "\"" + "." + func_name + "(" + "\"" + random_char_arg + "\"" + ")"
            else:
                stmt = "\"" + rand_str + "\"" + "." + func_name + "(" + "\"" + random_char_arg + "\"" + ", \"random\"" + ")"
        elif args == "Integer":
            int_arg = len(rand_str) * 2
            stmt = "\"" + rand_str + "\"" + "." + func_name + "(" + str(int_arg) + ")"
        else:
            stmt = "\"" + rand_str + "\"" + "." + func_name + "()"
    return stmt


def insert_var_to_rand_str(rand_str, var):
    insert_pos = int(np.random.randint(1, len(rand_str) - 2, 1))
    out_str = "(\"" + rand_str[:insert_pos] + "\"+ " + var + " + \"" + rand_str[insert_pos:] + "\")"
    return out_str


def generate_random_string(max_len=100):
    # includes lower and upper letters, digits, punctuation, and whitespace
    string_vocab = list(string.ascii_letters) + list(string.digits) + list(string.punctuation) + [" "]
    # remove certain characters that cause issues
    string_vocab.remove("#")
    string_vocab.remove("\"")
    string_vocab.remove("\'")
    string_vocab.remove("\\")
    string_vocab.remove("$")
    string_vocab.remove("`")

    str_len = int(np.random.randint(5, max_len, 1))
    random_seq = np.random.randint(0, len(string_vocab), str_len)
    out_str = ""
    for idx in random_seq:
        out_str += string_vocab[idx]
    out_str.replace("`", "")

    return out_str


def comp_block(max_no_stmts, variables, indentation, num_stmt_operands, libs,
               min_statement_operands, min_comp_block_size, string_funcs_df, lib_func_df_dict: Dict[str, Any],
               str_max_len=100):
    # do max_no_stmts + 1 since the upper bound is exclusive in randint
    no_stmts = int(np.random.randint(min_comp_block_size, max_no_stmts + 1, 1))
    statement_types = ["NUMERIC", "STRING"]
    lines = []
    imported_libs = []

    for i in range(0, no_stmts):
        stmt_type = statement_types[int(np.random.randint(0, len(statement_types), 1))]
        # do num_stmt_operands + 1 since the upper bound is exclusive in randint
        num_stmts = int(np.random.randint(min_statement_operands, num_stmt_operands + 1, 1))

        if stmt_type == "NUMERIC":
            stmt, imports = numeric_statement(get_lof_vars(variables, "NUMERIC"), num_stmts, libs,
                                              lib_func_df_dict=lib_func_df_dict)
            imported_libs += imports
        elif stmt_type == "STRING":
            stmt = string_statement(get_lof_vars(variables, "STRING"), string_funcs_df=string_funcs_df,
                                    max_len=str_max_len)

        assign = int(np.random.randint(0, 2, 1))  # virtual coin toss to decide if we do a variable assignment
        if assign == 0:
            rand_var_name = "var" + str(int(np.random.randint(0, 1000, 1)))
            stmt = get_indentation(indentation) + rand_var_name + " = " + stmt
        else:
            stmt = get_indentation(indentation) + stmt
        lines.append(stmt)

    return lines, imported_libs


def get_lof_vars(num_dict, type):
    out_lst = []
    for elem in num_dict[type]:
        out_lst.append(elem["colname"])
    return out_lst


def if_block(var_value_dict_list: List[Dict[str, Any]], indentation, if_else, if_list):
    # decide if we want to use variable vs variable or variable vs value
    # 0 = variable vs variable
    # 1 = variable vs value

    if len(var_value_dict_list) == 1:
        var_vs_var = 1  # there is no choice, compare variable with value
    else:
        var_vs_var = int(np.random.randint(0, 2, 1))
    if if_else:
        if var_vs_var == 0:
            while True:
                comp_var_name_1 = random.choice(var_value_dict_list)['colname']
                comp_var_name_2 = random.choice(var_value_dict_list)['colname']

                if comp_var_name_1 != comp_var_name_2:
                    break
            comp_operand = random.choice(COMP_OPS)
            line = get_indentation(indentation) + "if " + str(comp_var_name_1) + " " + comp_operand + " " + str(
                comp_var_name_2) + ":"

        else:
            while True:  # make sure that no variable is used twice in comparison
                comp_elem = random.choice(var_value_dict_list)
                comp_var_name = comp_elem["colname"]
                if comp_var_name not in if_list:
                    if_list.append(comp_var_name)
                    break
                if len(if_list) == len(var_value_dict_list):
                    break
            # select one percentile value from the list of percentiles for comparison
            comp_val = comp_elem["percentiles"][int(np.random.randint(0, len(comp_elem["percentiles"]), 1))]

            comp_operand = COMP_OPS[int(np.random.randint(0, len(COMP_OPS), 1))]
            line = get_indentation(indentation) + "if " + str(comp_var_name) + " " + comp_operand + " " + str(
                comp_val) + ":"

    else:  # if this bool is False, we are in the else case
        line = get_indentation(indentation) + "else:"
    return [line], if_list


def loop_block(indentation,
               max_iter=16):  # maximum number of iterations per loop that we allow; currently set to 16 to limit runtime
    type = int(np.random.randint(0, 2, 1))  # simulated coin-toss to decide on the loop type
    counter_var = "i" + str(int(np.random.randint(0, 10000,
                                                  1)))  # use this artificial var to avoid variable collisions when there are nested loops
    no_iter = int(np.random.randint(int(max_iter / 2), max_iter, 1))  # randomly choose the number of iterations

    if type == 0:  # if type==0 => for loop
        stmt = get_indentation(indentation) + "for " + counter_var + " in range(" + str(no_iter) + "):"
        return [stmt]
    else:  # if type==1 => while loop
        lines = []
        stmt = get_indentation(indentation) + counter_var + " = 0"
        lines.append(stmt)
        stmt = get_indentation(indentation) + "while " + counter_var + " < " + str(no_iter) + ":"
        lines.append(stmt)
        stmt = get_indentation(indentation + 1) + counter_var + " += 1"
        lines.append(stmt)
        return lines


def get_indentation(indentation):
    output = ""
    for i in range(0, indentation):
        output += "\t"
    return output


def get_weight_list(allowed_blocks, block_weights):
    out_list = []
    for block in allowed_blocks:
        out_list.append(block_weights[block])
    return out_list


def sample_blocks(max_no_childs: int, func_tree: Tree, if_count: int, max_if: int, allowed_blocks: List,
                  block_weights: Dict, to_visit: List, used_blocks: List, curr_node: str):
    no_childs = int(np.random.randint(1, max_no_childs + 1, 1))  # create a random number of children
    for i in range(0, no_childs):
        next_block = random.choices(allowed_blocks, weights=get_weight_list(allowed_blocks, block_weights), k=1)[0]
        used_blocks.append(next_block)
        if next_block == "IF" and if_count >= max_if:
            # iterate until a non-IF block gets sampled
            while next_block == "IF":  # choose a different block type
                next_block = allowed_blocks[int(np.random.randint(0, len(allowed_blocks), 1))]

        next_node = func_tree.create_node(next_block, parent=curr_node)

        # add block to queue if is not COMP
        if next_node.tag != "COMP":
            to_visit.append(next_node)

        # in case an IF block has been sampled, also create an ELSE block
        if next_block == "IF":
            if_count += 1
            next_node = func_tree.create_node("ELSE", parent=curr_node)
            to_visit.append(next_node)

    return if_count


def create_function_tree(max_childs, max_depth, allowed_blocks, max_if, exact_tree):
    # block weights according to the paper by Ramachandra
    # COMP blocks occur always, so their weighting is not relevant
    block_weights = {"COMP": 0.59, "LOOP": 0.07, "IF": 0.34}

    while True:  # iterate until we have a valid tree, e.g., the tree incorporates all allowed blocks
        func_tree = Tree()
        root = func_tree.create_node("START")
        to_visit = [root]
        if_count = 0
        used_blocks = []
        while len(to_visit) > 0:
            curr_node = to_visit.pop(0)
            if curr_node == root:
                if_count = sample_blocks(max_childs, func_tree, if_count=if_count, max_if=max_if,
                                         allowed_blocks=allowed_blocks, block_weights=block_weights, to_visit=to_visit,
                                         used_blocks=used_blocks, curr_node=curr_node)
            else:
                if func_tree.depth(
                        curr_node) < max_depth - 1:  # if we are at this level, we can create nodes that can have childs => IF and LOOPS
                    if curr_node.tag not in ["COMP", "ROOT"]:
                        if_count = sample_blocks(max_childs - 1, func_tree, if_count=if_count, max_if=max_if,
                                                 allowed_blocks=allowed_blocks, block_weights=block_weights,
                                                 to_visit=to_visit, used_blocks=used_blocks, curr_node=curr_node)
                else:  # if we are at the last level, we can only create compute nodes
                    # if we are at leave level, we can only create COMP nodes
                    # we do not need to append this to to_visit since we are at leaf level
                    no_childs = int(np.random.randint(1, max_childs, 1))
                    for i in range(0, no_childs):
                        func_tree.create_node("COMP", parent=curr_node)
        if exact_tree:
            # check if all node types have been used; if not then create a random function tree again
            if sorted(list(set(used_blocks))) == sorted(allowed_blocks):
                break
        else:
            break

    return func_tree, root, list(set(used_blocks))


def create_UDF(var_value_dict, libs, dbms: str, string_funcs_df, lib_func_df_dict: Dict[str, Any], exact_tree=False,
               max_comp_block_size=5, max_statement_operands=4,
               allowed_blocks=["COMP", "IF", "LOOP"],
               max_childs=3, max_depth=3, max_if=3, ret_var=None, incol_list=None, tree=None, root=None,
               min_comp_block_size=1, min_statement_operands=1):
    to_visit = tree.children(root.identifier)
    lines = []
    imports = []
    var_list = []

    # if_list: keep track of variables that were already used in if_statements
    if_list = []

    # if we generate the queries with Benjamins generator, we need to use the incol_list since this is the correct order
    # when we use the var_value_dict, the order is not correct

    if incol_list is None:
        # create header of the function
        var_str = ""
        for elem in var_value_dict.keys():
            for var in var_value_dict[elem]:
                var_list.append(var["colname"])
                var_str += str(var["colname"] + ", ")
        var_str = var_str[:-2]

        if dbms == 'postgres':
            curr_str = "def test_func(" + var_str + "):"
            lines.append(curr_str)
    else:
        for elem in incol_list:
            var_list.append(elem)

        if dbms == 'postgres':
            curr_str = "def test_func(" + ", ".join(incol_list) + "):"
            lines.append(curr_str)

    while len(to_visit) > 0:
        curr_node = to_visit.pop(0)
        if curr_node.tag == "COMP":
            curr_lines, imported = comp_block(max_comp_block_size, var_value_dict, tree.depth(curr_node),
                                              max_statement_operands,
                                              min_statement_operands=min_statement_operands,
                                              min_comp_block_size=min_comp_block_size, libs=libs,
                                              string_funcs_df=string_funcs_df, lib_func_df_dict=lib_func_df_dict)
            imports += imported
        elif curr_node.tag == "IF":
            curr_lines, if_list = if_block(var_value_dict["NUMERIC"], tree.depth(curr_node), True, if_list)
        elif curr_node.tag == "ELSE":
            curr_lines, if_list = if_block(var_value_dict["NUMERIC"], tree.depth(curr_node), False, if_list)
        elif curr_node.tag == "LOOP":
            curr_lines = loop_block(tree.depth(curr_node))
        else:
            raise Exception(f'{curr_node.tag}')

        lines = lines + curr_lines

        # add children of this node to the top of the queue
        if tree.children(curr_node.identifier) != []:
            to_visit = tree.children(curr_node.identifier) + to_visit

    if ret_var is None:
        ret_var = var_list[int(np.random.randint(0, len(var_list), 1))]

    stmt = get_indentation(1) + "return " + ret_var  # function returns one of the input variables
    lines.append(stmt)

    # add a tab in front of each element in imports
    imports = [get_indentation(1) + x for x in imports]
    if dbms == 'postgres':
        lines = lines[0:1] + list(set(imports)) + list(lines[1:])
    elif dbms == 'duckdb':
        # skip imports since libraries should be loaded already into the environment
        # lines = list(set(imports)) + list(lines)
        pass

    return lines, ret_var
