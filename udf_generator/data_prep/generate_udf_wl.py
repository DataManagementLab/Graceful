import functools
import json
import multiprocessing
import random
import sys
import traceback
from typing import List, Tuple

import pandas as pd

from data_prep.UDFfuzz.UDF_fuzz import create_function_tree, create_UDF
from data_prep.utilities import create_data_type_dict, db_to_dict, get_func_call, write_to_file
from utils.utils import t_print

sys.path.append("..")
import os
from tqdm import tqdm


def create_SQL_func(UDF_lines, funcname, var_value_dict, ret_var, dbms: str, incols=None):
    def get_col_dtype_map(var_value_dict, incols):
        col_dtype_map = []

        for col in incols:
            for elem in var_value_dict.keys():
                for var in var_value_dict[elem]:
                    if col == var["colname"]:
                        col_dtype_map.append((col, var["datatype"]))
                        break
        return col_dtype_map

    lines = []
    if dbms == "postgres":
        header = "CREATE OR REPLACE FUNCTION " + funcname + "("
    elif dbms == "duckdb":
        header = "def " + funcname + "("
    else:
        raise NotImplementedError(f'Unknown dbms: {dbms}')

    def duckdb_translate(dtype: str) -> str:
        translate_dict = {'integer': 'int', 'bigint': 'int', 'varchar': 'str', 'double': 'float',
                          'double precision': 'float', 'char': 'str',
                          'float': 'float', 'date': 'date', 'time': 'time', 'decimal': 'float'}
        if dtype.lower().startswith('decimal'):
            return 'float'
        else:
            return translate_dict[dtype.lower()]

    duckdb_in_dtype = []
    if incols is None:
        vars = []
        for elem in var_value_dict.keys():
            for var in var_value_dict[elem]:
                if var['datatype'] is None:
                    print(f'None datatype for {var}')
                    continue

                if dbms == 'postgres':
                    header += "\"" + str(var["colname"]) + "\"" + " " + str(var["datatype"]) + ","
                elif dbms == 'duckdb':
                    header += f'{var["colname"]}:{duckdb_translate(var["datatype"])},'
                vars.append(var["colname"])
        header = header[:-1]  # remove the last comma
    else:
        for col in get_col_dtype_map(var_value_dict, incols):
            if dbms == 'postgres':
                header += "\"" + col[0] + "\"" + " " + col[1] + ","
            elif dbms == 'duckdb':
                header += f'{col[0]}:{duckdb_translate(col[1])},'
                dtype = col[1].upper()
                if dtype.startswith('DECIMAL'):
                    dtype = 'FLOAT'
                duckdb_in_dtype.append(dtype)
        header = header[:-1]  # remove the last comma
        vars = incols
    ret_dt = None
    # not the smartest way to retrieve the datatype of the return type but don't care for now
    for elem in var_value_dict.keys():
        if ret_dt is not None:
            break
        for var in var_value_dict[elem]:
            if ret_var == var["colname"]:
                ret_dt = var["datatype"]
                break
    assert ret_dt is not None, f'No return datatype found for {ret_var}:\n{var_value_dict}'

    if dbms == 'postgres':
        header += ") RETURNS " + str(ret_dt)
    elif dbms == 'duckdb':
        header += f') -> {duckdb_translate(ret_dt)}:'

    lines.append(header)
    if dbms == 'postgres':
        lines.append("AS $$")

    for line in UDF_lines:
        line_tab = line  # insert UDF code and add one tab before each line
        lines.append(line_tab)

    if dbms == 'postgres':
        if incols is None:
            # call the name of the UDF and return it
            return_str = "return test_func("  # in the fuzzer we name the function always "test_func"
            for elem in var_value_dict.keys():
                for var in var_value_dict[elem]:
                    return_str += str(var["colname"]) + ","
            return_str = return_str[:-1] + ")"
        else:
            return_str = "return test_func("  # in the fuzzer we name the function always "test_func"
            for elem in incols:
                return_str += elem + ","
            return_str = return_str[:-1] + ")"
        lines.append(return_str)
        lines.append("$$ LANGUAGE plpython3u;")
    elif dbms == 'duckdb':
        # create function with explicit type casts
        ret_dt = ret_dt.upper()
        if ret_dt.startswith('DECIMAL'):
            ret_dt = 'FLOAT'
        lines.append(f'db_conn.create_function(\'{funcname}\',{funcname},{duckdb_in_dtype},{ret_dt.upper()})')
    return lines, vars


def _randomly_pick_table(database, max_params: int, try_to_produce_two_numeric_params: bool = False):
    if len(database.keys()) > 1:
        rand_tab = random.choice(list(database.keys()))
    elif len(database.keys()) == 1:
        rand_tab = list(database.keys())[0]
    else:
        print(database)
        return None, None
    # from the table, a random number of columns that are used as params for the UDF
    num_cols_in_table = len(database[rand_tab])
    assert num_cols_in_table > 0, num_cols_in_table

    # pick randomly between 1 and max_params columns
    # try to produce at least two numeric parameters in case flag is set
    # this is important for the branch case, so that we later on can have an if statement comparing two columns
    input_params = None
    for _ in range(10):
        col_names = random.sample(database[rand_tab], k=int(random.randint(1, min(num_cols_in_table, max_params))))
        input_params = create_data_type_dict(col_names)

        if try_to_produce_two_numeric_params:
            if len(input_params["NUMERIC"]) >= 2:
                break
        else:
            break

    return rand_tab, input_params


def create_drop_str(UDF_name: str, dbms: str):
    if dbms == 'postgres':
        return f'DROP FUNCTION {UDF_name};'
    elif dbms == 'duckdb':
        return f'db_conn.drop_function(\'{UDF_name}\')'
    else:
        raise NotImplementedError(f'Unknown dbms: {dbms}')


def SQL_function_generator(task: Tuple[str, int, str], no_funcs: int, exact_tree, loc, dbms: str, db_metadata_path: str,
                           max_params=20,
                           branch=True, ):
    """

    :param no_funcs: only relevant when not passing in a meta_file
    :param db_name:
    :param global_counter:
    :param exact_tree:
    :param loc:
    :param max_params:
    :param branch:
    :param meta_file:
    :return:
    """
    try:
        db_name, global_counter, meta_file = task
        t_print(f'Generating UDFs for database {db_name} (exact tree: {exact_tree}, branch: {branch}).', flush=True)

        calls = []
        drop_funcs = []
        SQL_lines_list = []
        func_table_dict = {}
        code_dict = {}
        runner_var = global_counter
        # Hyperparams for UDF fuzzer
        max_childs = 2
        max_depth = 3
        max_if = 3
        # Use two approaches for shifting the testing set
        # 1. 100-150% of the original/default max
        # 2. 150-250% of the original/default max
        min_comp_block_size = 9  # Default: 1 | 1: 6 | 2: 9
        max_comp_block_size = 15  # Default: 6 | 1: 9 | 2: 15
        min_statement_operands = 5  # Default: 1 | 1: 5 | 2: 7
        max_statement_operands = 5  # Default: 5 | 1: 7 | 2: 12
        # Exp 3.1.1: Comp block size between 6 and 9, statement operands between 5 and 7 (MODERATE)
        # Exp 3.1.2: Comp block size between 9 and 15, statement operands between 7 and 12 (AGGRESSIVE)
        allowed_blocks = ["COMP", "LOOP", "IF"]

        branch_only_on_two_numeric_col_tables = False

        database_branch_dict = dict(
            (b, db_to_dict(db_name, min_table_size=1, threshold=0.2,
                           path=db_metadata_path, branch=b,
                           meta_file=meta_file,
                           branch_only_on_two_numeric_col_tables=branch_only_on_two_numeric_col_tables)) for b in
            [True, False])

        # load string funcs to pandas dataframe
        string_funcs_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'UDFfuzz', "string_funcs.csv"))

        # load lib functions to pandas dataframe
        lib_func_df_dict = dict()
        for lib in ["numpy", "math"]:
            lib_func_df_dict[lib] = pd.read_csv(
                os.path.join(os.path.dirname(__file__), 'UDFfuzz', 'libs', lib + ".csv"))

        if meta_file is None:
            for i in tqdm(range(0, no_funcs)):
                tree, root, used_blocks = create_function_tree(allowed_blocks=allowed_blocks, max_childs=max_childs,
                                                               max_depth=max_depth, max_if=max_if,
                                                               exact_tree=exact_tree)
                # see if the randomly created udf structure contains a branch since this determines the table we can use
                # as we need at least two numerical columns
                if "IF" in used_blocks:
                    branch = True

                database = database_branch_dict[branch]

                UDF_name = "func_" + str(runner_var)
                runner_var += 1
                drop_funcs.append(create_drop_str(UDF_name, dbms))  # already prepare statement for removing the UDFs

                # when a branch occurs, we have to make sure that we have at least two numerical input variables
                if branch:
                    input_params = None
                    rand_tab = None
                    # iterate until there are at least two numeric parameters
                    break_for = False

                    if branch_only_on_two_numeric_col_tables:
                        numeric_cols = 2
                    else:
                        numeric_cols = 1

                    while input_params is None or len(input_params["NUMERIC"]) < numeric_cols:
                        # pick randomly one table from the database
                        rand_tab, input_params = _randomly_pick_table(database, max_params=max_params,
                                                                      try_to_produce_two_numeric_params=True)
                        if input_params is None:
                            # no table is available!
                            break_for = True
                            break

                    if break_for:
                        print(f'Break for')
                        break
                else:
                    # pick randomly one table from the database
                    rand_tab, input_params = _randomly_pick_table(database, max_params=max_params)

                func_table_dict[UDF_name] = rand_tab

                UDF_lines, ret_var = create_UDF(var_value_dict=input_params, libs=["numpy", "math"],
                                                max_comp_block_size=max_comp_block_size,
                                                max_statement_operands=max_statement_operands, tree=tree, root=root,
                                                min_comp_block_size=min_comp_block_size,
                                                min_statement_operands=min_statement_operands, dbms=dbms,
                                                string_funcs_df=string_funcs_df, lib_func_df_dict=lib_func_df_dict)
                SQL_lines, vars = create_SQL_func(UDF_lines, UDF_name, input_params, ret_var, dbms=dbms)

                # use the header of the UDF definition to create the call
                calls.append(get_func_call(UDF_name, vars, rand_tab))
                # append UDFs to a long .sql file which we then use to insert all the data to the database
                # write_to_file(SQL_lines, os.path.join(loc, db_name, 'created_code'), UDF_name, mode="w", extension=".txt")
                code_dict[UDF_name] = SQL_lines
                SQL_lines_list.extend(SQL_lines)

            write_to_file(calls, os.path.join(loc, db_name, 'sql_scripts'), "calls", mode="w", extension=".sql")

        else:
            def get_colnames_dict_from_lst(incols, database, table):
                col_names = []
                for elem in database[table]:
                    if elem["colname"] in incols:
                        col_names.append(elem)
                return col_names

            def branch_possible(in_params):
                if len(in_params["NUMERIC"]) <= 1:
                    return False
                else:
                    return True

            code_dict = {}

            database = db_to_dict(db_name, min_table_size=1, threshold=0.2, path=db_metadata_path, branch=branch,
                                  meta_file=meta_file)
            # load json stored in meta_file
            with open(meta_file, "r") as f:
                meta = json.load(f)
            for func in meta.keys():

                UDF_name = func
                # when a branch occurs, we have to make sure that we have at least two numerical input variables
                if branch:
                    rand_tab = meta[func]["table"]
                    if database.get(rand_tab) is None:
                        print(f'Table {rand_tab} not found in database {db_name}. Skipping function {func}.')
                        continue
                    col_names = get_colnames_dict_from_lst(meta[func]["incols"], database, rand_tab)
                    input_params = create_data_type_dict(col_names)

                    if exact_tree:
                        if branch_possible(input_params):
                            func_table_dict[UDF_name] = rand_tab
                            drop_funcs.append(
                                create_drop_str(UDF_name, dbms))  # already prepare statement for removing the UDFs

                            tree, root, used_blocks = create_function_tree(allowed_blocks=["COMP", "LOOP", "IF"],
                                                                           max_childs=max_childs, max_depth=max_depth,
                                                                           max_if=max_if, exact_tree=exact_tree)
                            UDF_lines, ret_var = create_UDF(input_params, ["numpy", "math"],
                                                            max_comp_block_size=max_comp_block_size,
                                                            max_statement_operands=max_statement_operands,
                                                            ret_var=meta[func]["outcol"], dbms=dbms,
                                                            incol_list=meta[func]["incols"], tree=tree, root=root,
                                                            string_funcs_df=string_funcs_df,
                                                            lib_func_df_dict=lib_func_df_dict)
                            SQL_lines, vars = create_SQL_func(UDF_lines, UDF_name, input_params,
                                                              ret_var=meta[func]["outcol"], incols=meta[func]["incols"],
                                                              dbms=dbms)
                        else:
                            print(f'Exact tree no branch possible for function {func}. Skipping function {func}.')
                            continue
                    else:
                        if branch_possible(input_params):
                            func_table_dict[UDF_name] = rand_tab

                            drop_funcs.append(
                                create_drop_str(UDF_name, dbms))  # already prepare statement for removing the UDFs
                            tree, root, used_blocks = create_function_tree(allowed_blocks=["COMP", "LOOP", "IF"],
                                                                           max_childs=max_childs, max_depth=max_depth,
                                                                           max_if=max_if, exact_tree=exact_tree)
                            UDF_lines, ret_var = create_UDF(input_params, ["numpy", "math"],
                                                            max_comp_block_size=max_comp_block_size,
                                                            max_statement_operands=max_statement_operands,
                                                            ret_var=meta[func]["outcol"], dbms=dbms,
                                                            incol_list=meta[func]["incols"], tree=tree, root=root,
                                                            string_funcs_df=string_funcs_df,
                                                            lib_func_df_dict=lib_func_df_dict)
                            try:
                                SQL_lines, vars = create_SQL_func(UDF_lines, UDF_name, input_params,
                                                                  ret_var=meta[func]["outcol"],
                                                                  incols=meta[func]["incols"],
                                                                  dbms=dbms)
                            except Exception as e:
                                print(f'{meta[func]}, {input_params}, {db_name}', flush=True)
                                raise e
                        else:
                            func_table_dict[UDF_name] = rand_tab
                            drop_funcs.append(
                                create_drop_str(UDF_name, dbms))  # already prepare statement for removing the UDFs
                            tree, root, used_blocks = create_function_tree(allowed_blocks=["COMP", "LOOP"],
                                                                           max_childs=max_childs, max_depth=max_depth,
                                                                           max_if=max_if, exact_tree=exact_tree)
                            UDF_lines, ret_var = create_UDF(input_params, ["numpy", "math"],
                                                            max_comp_block_size=max_comp_block_size,
                                                            max_statement_operands=max_statement_operands,
                                                            ret_var=meta[func]["outcol"], dbms=dbms,
                                                            incol_list=meta[func]["incols"], tree=tree, root=root,
                                                            string_funcs_df=string_funcs_df,
                                                            lib_func_df_dict=lib_func_df_dict)
                            try:
                                SQL_lines, vars = create_SQL_func(UDF_lines, UDF_name, input_params,
                                                                  ret_var=meta[func]["outcol"],
                                                                  incols=meta[func]["incols"],
                                                                  dbms=dbms)
                            except Exception as e:
                                print(f'{meta[func]}, {input_params}, {db_name}', flush=True)
                                raise e

                else:
                    rand_tab = meta[func]["table"]
                    if database.get(rand_tab) is None:
                        continue
                    col_names = get_colnames_dict_from_lst(meta[func]["incols"], database, rand_tab)
                    input_params = create_data_type_dict(col_names)
                    func_table_dict[UDF_name] = rand_tab
                    drop_funcs.append(
                        create_drop_str(UDF_name, dbms))  # already prepare statement for removing the UDFs
                    tree, root, used_blocks = create_function_tree(allowed_blocks=["COMP", "LOOP"],
                                                                   max_childs=max_childs, max_depth=max_depth,
                                                                   max_if=max_if, exact_tree=exact_tree)
                    UDF_lines, ret_var = create_UDF(input_params, ["numpy", "math"],
                                                    max_comp_block_size=max_comp_block_size,
                                                    max_statement_operands=max_statement_operands,
                                                    ret_var=meta[func]["outcol"], dbms=dbms,
                                                    incol_list=meta[func]["incols"], tree=tree, root=root,
                                                    string_funcs_df=string_funcs_df, lib_func_df_dict=lib_func_df_dict)
                    SQL_lines, vars = create_SQL_func(UDF_lines, UDF_name, input_params, ret_var=meta[func]["outcol"],
                                                      incols=meta[func]["incols"], dbms=dbms)

                SQL_lines_list.extend(SQL_lines)
                # write_to_file(SQL_lines, os.path.join(loc, db_name, 'created_code'), UDF_name, mode="w", extension=".txt")
                code_dict[UDF_name] = SQL_lines

        assert len(
            code_dict) > 0, f'No UDFs generated for database {db_name} (exact tree: {exact_tree}, branch: {branch}).'
        os.makedirs(os.path.join(loc, db_name, 'sql_scripts'), exist_ok=True)
        with open(os.path.join(loc, db_name, 'sql_scripts', 'udfs.json'), 'w') as f:
            json.dump(code_dict, f)

        write_to_file(SQL_lines_list, os.path.join(loc, db_name, 'sql_scripts'), "udfs", mode="w",
                      extension=".sql")
        write_to_file(drop_funcs, os.path.join(loc, db_name, 'sql_scripts'), "dropUDFs", mode="w", extension=".sql")
        # create a dataframe from func_table_dict to save it as a csv file
        df = pd.DataFrame.from_dict(func_table_dict, orient="index")
        df.to_csv(os.path.join(loc, db_name, "func_table_dict.csv"))
        return runner_var
    except Exception as e:
        print(e)
        traceback.print_exc()
        raise e


def dataset_generator(database_list: List[str], exp_folder_path: str, queries_per_db, with_branch, exact_tree,
                      dbms: str,
                      db_metadata_path: str, meta_file=False):
    tasks = []

    fn = functools.partial(SQL_function_generator, no_funcs=queries_per_db, exact_tree=exact_tree, branch=with_branch,
                           loc=exp_folder_path, db_metadata_path=db_metadata_path, dbms=dbms)

    if not meta_file:
        global_counter = 0
        for db in tqdm(database_list):
            tasks.append((db, global_counter, None))
            global_counter += queries_per_db
            # global_counter = SQL_function_generator(queries_per_db, exact_tree=exact_tree, branch=with_branch,
            #                                         db_name=db, global_counter=global_counter, loc=exp_folder_path,
            #                                         dbms=dbms)
    else:
        for db in tqdm(database_list):
            tasks.append((db, 0, os.path.join(exp_folder_path, db, str(meta_file))))
            # SQL_function_generator(queries_per_db, exact_tree=exact_tree, branch=with_branch, db_name=db,
            #                        global_counter=0, loc=exp_folder_path,
            #                        meta_file=os.path.join(exp_folder_path, db, str(meta_file)), dbms=dbms)
    t_print(f'Spawn tasks: {len(tasks)}', flush=True)
    with multiprocessing.Pool() as executor:
        executor.map(fn, tasks)


def setup_exp_folder(database_lst, folder_loc: str, exp_folder_name: str):
    # create the folder for the experiment
    path = os.path.join(folder_loc, exp_folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    # create the folder that stores the sub_folders for the dbs
    if not os.path.exists(os.path.join(path, "dbs")):
        os.makedirs(os.path.join(path, "dbs"))
    # create the folder for each database
    for db in database_lst:
        if not os.path.exists(os.path.join(path, "dbs", db)):
            os.makedirs(os.path.join(path, "dbs", db))
        # for each database folder, create two subfolders named "created_code" and "created_graphs"
        if not os.path.exists(os.path.join(path, "dbs", db, "created_code")):
            os.makedirs(os.path.join(path, "dbs", db, "created_code"))
        if not os.path.exists(os.path.join(path, "dbs", db, "sql_scripts")):
            os.makedirs(os.path.join(path, "dbs", db, "sql_scripts"))
    return os.path.join(path, "dbs")
