import os

import pandas as pd


def write_to_file(lines, folder, filename, mode, extension):
    """
    Function takes the lines of a UDF as input and writes it to a .py file at location /data/created_code
    """
    location = os.path.join(folder, filename + extension)
    os.makedirs(os.path.dirname(location), exist_ok=True)
    with open(location, mode, encoding="utf-8") as file:
        for line in lines:
            file.write(line)
            file.write("\n")


def get_func_call(func_name, vars, table):
    # use regex to extract content from body
    # UDF_regex = re.compile('func_((\d*)\([\w,\s]*\))')
    # UDF_match = UDF_regex.search(udf_header)
    # udf_params = str(UDF_match.group(0)).split("(")[1][:-1]

    var_str = ""
    for var in vars:
        var_str += "\"" + var + "\"" + ","
    var_str = var_str[:-1]
    return "SELECT " + func_name + "(" + var_str + ")" + " FROM " + "\"" + table + "\"" + ";"


def db_to_dict(db_name, min_table_size, threshold, path, branch=False, meta_file=None,
               branch_only_on_two_numeric_col_tables=False):
    """
    Helper function that extracts information about a database from the database metadata file
    These information consists of tables of the dabe with corresponding columns and their corresponding datatypes
    """
    # read csv from path to a dataframe
    df = pd.read_csv(path)
    # only keep the rows where the db is equal to the db_name
    df = df[df['db'] == db_name]
    # only keep the rows where use is YES
    df = df[df['use'] == 'YES']
    # only keep the rows where the nan_rato is smaller than threshold and where nan_ratio is not nan
    df = df[(df['nan_ratio'] < threshold) & (df['nan_ratio'].notna())]
    # only keep the rows where the table_rows is bigger than 0
    df = df[df['table_rows'] > 0]

    # if there is a branch, make sure that there are enough numerical values in the table
    # also check if there is a meta_file
    # if we have a meta file, the query that are generated might not have enough numerical values that are needed for branching
    if branch and meta_file is None:
        if branch_only_on_two_numeric_col_tables:
            df = df[df["no_num_values"] >= 2]
        else:
            df = df[df["no_num_values"] >= 1]

    assert len(
        df) > 0, f'no rows left in dataframe {df} / {db_name} / {path} / {threshold} / {branch} / {meta_file} / {threshold} / {min_table_size}'
    out_dict = {}
    for index, row in df.iterrows():
        # create a dict where the key is the column name and the value is the datatype
        col_dict = {}
        col_dict["PK"] = row['PK']
        col_dict["colname"] = row["column"]
        col_dict["datatype"] = row["datatype"]
        col_dict["percentiles"] = [row[f"percentile_{i}"] for i in range(0, 100, 10)]
        if col_dict["datatype"].lower() in ["integer", "double precision", "float", "decimal", "bigint", 'double'] or \
                col_dict[
                    'datatype'].startswith('DECIMAL'):
            col_dict["category"] = "NUMERIC"
        elif col_dict["datatype"].lower() in ["char", "varchar"]:
            col_dict["category"] = "STRING"
        elif col_dict["datatype"].lower() in ["date"]:
            col_dict["category"] = "DATE"
        elif col_dict['datatype'].lower() in ['time']:
            col_dict["category"] = "TIME"
        else:
            raise Exception(f"Unknown datatype {col_dict['datatype']}")

        if row['table'] in out_dict:
            out_dict[row['table']].append(col_dict)
        else:
            out_dict[row['table']] = [col_dict]
    # remove all tables from the database that have only a limited number of columns
    for table in list(out_dict.keys()):
        if len(out_dict[table]) < min_table_size:
            del out_dict[table]
    return out_dict


def create_data_type_dict(selected_cols):
    """
    Function takes a list of randomly selected columns from a table as input
    It analyzes this input and assigns each dict to certain bucket based on its data type category (e.g., STRING, NUMERIC)
    """
    out_dict = {
        "NUMERIC": [],
        "STRING": [],
        "DATE": [],
        "TIME": []
    }
    for elem in selected_cols:
        if "category" in elem.keys():
            if elem["category"] == "NUMERIC":
                out_dict["NUMERIC"].append(elem)
            elif elem["category"] == "STRING":
                out_dict["STRING"].append(elem)
            elif elem["category"] == "DATE":
                out_dict["DATE"].append(elem)
            elif elem["category"] == "TIME":
                out_dict["DATE"].append(elem)
            else:
                print(f"Unknown category: {elem['category']}")
        # Might expand with other categories later on
        else:
            print(f"Unknown category: {elem}")
    return out_dict


def update_table_sizes(meta_data_path: str, data_path: str):
    folders = ["financial", "accidents", "scaled_basketball", "employee", "geneea", "scaled_fhnk"]
    # read the csv file at meta_data_path to a dataframe
    meta_data = pd.read_csv(meta_data_path, sep=",")
    for folder in folders:
        folder_path = os.path.join(data_path, folder)
        # iterate over all csv.files in folder_path
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                table_name = file.split(".")[0]
                # determine the number of rows in the csv file
                df = pd.read_csv(os.path.join(folder_path, file), sep="\t")
                # update the number of rows in the meta_data dataframe
                # update the rows in metadata where column "db" is folder and column table is table_name
                meta_data.loc[(meta_data["db"] == folder) & (meta_data["table"] == table_name), "table_rows"] = len(
                    df.index)

    # write the updated meta_data dataframe to the meta_data_path
    meta_data.to_csv(meta_data_path, index=False)
