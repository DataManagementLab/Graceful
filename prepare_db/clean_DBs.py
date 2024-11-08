import argparse
import os

import psycopg2

from prepare_db.utilities import db_to_dict


def create_SQL_file(db_name, table_col_map, work_dir):
    """
    Helper function that creates a .sql file which includes the commands for removing NULL rows and
    for avoiding 0 rows by adding 1 to all numerical columns
    """

    # create a new file with the name db_name + "_clean_up.sql"
    file = open(os.path.join(work_dir, 'cleanDB_scripts', f'{db_name}_clean_up.sql'), "w")

    # iterate over all tables in the list tables
    for table in sorted(list(table_col_map.keys())):
        delete_cols_num = []
        delete_cols_str = []
        update_cols = []
        for col in table_col_map[table]:
            # check if key "category" exists in the dictionary col
            if "category" in col.keys():
                # if numeric
                if col["category"] == "NUMERIC":
                    # if the column is primary key, then we delete the rows with NULL or 0 values
                    if col["PK"] == 1:
                        delete_cols_num.append(col["colname"])
                    else:
                        # if the column is not the primary key, then we will add 1 to all values of the column to avoid 0s
                        # we also delete the rows with NULL values
                        delete_cols_str.append(
                            col["colname"])  # use here delete_cols_str since we only want to delete NULL

                        col_whitelist = [
                            ('financial', 'district', 'A13'),
                            # has no zero values and will otherwise exceed the dtype width
                        ]
                        if (db_name, table, col["colname"]) in col_whitelist:
                            pass
                        else:
                            update_cols.append(col["colname"])
                # if string type (char or varchar)
                if col["category"] == "STRING":
                    delete_cols_str.append(col["colname"])
        delete_str_num = "DELETE FROM " + "\"" + table + "\"" + " WHERE "
        delete_str_str = "DELETE FROM " + "\"" + table + "\"" + " WHERE "
        update_str = "UPDATE " + "\"" + table + "\"" + " SET " + "\""
        for col in delete_cols_num:
            delete_str_num += "\"" + col + "\"" + " IS NULL OR " + "\"" + col + "\"" + " = 0 OR "
        for col in delete_cols_str:
            delete_str_str += "\"" + col + "\"" + " IS NULL OR "
        for col in update_cols:
            update_str += col + "\"" + " = " + "\"" + col + "\"" + " + 1, " + "\""
        if len(delete_cols_num) > 0:
            delete_str_num = delete_str_num[:-4] + ";" + "\n"
            file.write(delete_str_num)
        if len(delete_cols_str) > 0:
            delete_str_str = delete_str_str[:-4] + ";" + "\n"
            file.write(delete_str_str)
        if len(update_cols) > 0:
            update_str = update_str[:-3] + ";" + "\n"
            file.write(update_str)


def exec(cursor, conn, dbms, sql):
    if dbms == 'postgres':
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.commit()
    elif dbms == 'duckdb':
        result = cursor.execute(sql).fetchall()
    else:
        raise NotImplementedError(f'Unknown dbms: {dbms}')
    return result


def perform_clean_up(db_name, tables, dbms: str, dbms_kwargs: dict, work_dir: str):
    """
    Helper function that reads a .sql file for each database and triggers the execution of that SQL file
    """
    card_after = {}

    if dbms == 'postgres':
        # Connect to your postgres DB
        conn = psycopg2.connect(database=db_name, host="localhost", user="postgres", port=dbms_kwargs['port'],
                                password=dbms_kwargs['password'],
                                options='-c statement_timeout=300000')
        # Open a cursor to perform database operations
        cur = conn.cursor()
    elif dbms == 'duckdb':
        import duckdb
        print(f'Duckdb version: {duckdb.__version__}')
        version = duckdb.__version__
        if version == '0.9.2':
            version_str = ''
        elif version == '0.10.1':
            version_str = '_10_1'
        else:
            raise ValueError(f"Unknown version {version}")
        db_path = f'{dbms_kwargs["dir"]}/{db_name}{version_str}.db'
        assert os.path.exists(db_path), f'Database {db_path} does not exist'
        conn = duckdb.connect(database=db_path)
        cur = conn
    else:
        raise NotImplementedError(f'Unknown dbms: {dbms}')

    file = os.path.join(work_dir, 'cleanDB_scripts', f'{db_name}_clean_up.sql')
    # check if the file size is greater 0
    if os.stat(file).st_size > 0:
        sql = open(file, 'r').read()
        try:
            exec(cur, conn, dbms, sql)
        except Exception as e:
            print(db_name)
            print(sql)
            raise e
    cur.execute('VACUUM ANALYZE;')

    for table in tables:
        # Execute a query before clean up
        sql = 'SELECT count(*) FROM ' + '\"' + table + '\"'

        card_after[table] = exec(cur, conn, dbms, sql)[0][0]
    return card_after


def clean_up_DBs(db: str, dbms: str, dbms_kwargs: dict, work_dir: str, nan_threshold=0.2, ):
    """
    Wrapper function that wraps the two helper functions from above
    So first, for each database the .sql file for cleaning is created
    Second, this cleaning file is executed in the database to clean it up
    """
    metadata_path = os.path.join(work_dir, 'db_metadata.csv')

    FLAG_path = os.path.join(work_dir, f'FLAG_CLEANED_{db}')
    if os.path.exists(FLAG_path):
        print(f'{db} already cleaned')
        return

    try:
        # filter the meta_data dataframe for the database db
        # the 2nd parameter describes minimum number of columns for a table
        db_dict = db_to_dict(db, 1, nan_threshold, metadata_path)
        create_SQL_file(db, db_dict, work_dir=work_dir)
        card_after = perform_clean_up(db, db_dict.keys(), dbms=dbms, dbms_kwargs=dbms_kwargs, work_dir=work_dir)

        fp = open(FLAG_path, 'w')
        fp.close()
    except Exception as e:
        print(db)
        print(e)
        raise e


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
