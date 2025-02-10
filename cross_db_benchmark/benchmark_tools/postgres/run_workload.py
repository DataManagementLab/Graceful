import datetime
import gc
import json
import os
import random
import re
import time
import traceback
from json.decoder import JSONDecodeError
from pathlib import Path

import duckdb.typing
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.load_database import create_db_conn
from cross_db_benchmark.benchmark_tools.postgres.check_valid import check_valid
from cross_db_benchmark.benchmark_tools.utils import load_json, pullup_udf_in_sql, rewrite_query_to_schema
from utils.utils import t_print

column_regex = re.compile('"(\S+)"."(\S+)"')


def extract_columns(sql):
    return [m for m in column_regex.findall(sql)]


def index_creation_deletion(existing_indexes, sql_part, db_conn, timeout_sec):
    cols = extract_columns(sql_part)
    index_cols = set(existing_indexes.keys())
    no_idxs = len(index_cols.intersection(cols))

    if len(cols) > 0:
        # not a single index for sql available, create one
        if no_idxs == 0:
            t, c = random.choice(cols)
            db_conn.set_statement_timeout(10 * timeout_sec, verbose=False)
            print(f"Creating index on {t}.{c}")
            index_creation_start = time.perf_counter()
            try:
                index_name = db_conn.create_index(t, c)
                existing_indexes[(t, c)] = index_name
                print(f"Creation time: {time.perf_counter() - index_creation_start:.2f}s")
            except Exception as e:
                print(f"Index creation failed {str(e)}")
            db_conn.set_statement_timeout(timeout_sec, verbose=False)

        # indexes for all columns, delete one
        if len(cols) > 1 and no_idxs == len(cols):
            t, c = random.choice(cols)
            print(f"Dropping index on {t}.{c}")
            try:
                index_name = existing_indexes[(t, c)]
                db_conn.drop_index(index_name)
                del existing_indexes[(t, c)]
            except Exception as e:
                print(f"Index deletion failed {str(e)}")


def modify_indexes(db_conn, sql_query, existing_indexes, timeout_sec):
    try:
        if 'GROUP BY ' in sql_query:
            sql_query = sql_query.split('GROUP BY ')[0]
        join_part = sql_query.split(" FROM ")[1].split(" WHERE ")[0]
        where_part = sql_query.split(" FROM ")[1].split(" WHERE ")[1]

        index_creation_deletion(existing_indexes, join_part, db_conn, timeout_sec)
        index_creation_deletion(existing_indexes, where_part, db_conn, timeout_sec)
    except Exception as e:
        print(f"Could not create indexes for {sql_query} ({str(e)})")


def run_pg_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path, run_kwargs,
                    repetitions_per_query, timeout_sec, with_indexes=False, cap_workload=None, hints=None,
                    min_runtime=100, no_variants=26, udf_load_path: str = None, udf_drop_path: str = None,
                    pullup_udf: bool = False, pullup_udf_intermed: bool = False):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if udf_load_path is not None and database == DatabaseSystem.POSTGRES:
        import subprocess
        assert udf_drop_path is not None
        # create the plpython3u extension
        return_code = subprocess.run(
            f'psql -d ${db_name} -h localhost -p ${database_conn_args["port"]} -U ${database_conn_args["user"]} -c "create extension plpython3u;"',
            shell=True)
        t_print(f'Load UDFs library exited with code: {return_code}')

        # register the UDFs
        return_code = subprocess.run(
            f'psql -d ${db_name} -h localhost -p ${database_conn_args["port"]} -U ${database_conn_args["user"]} -c "\i ${udf_load_path}"',
            shell=True)
        t_print(f'Load UDFs exited with code: {return_code}')

    dbms_wrapper = create_db_conn(database, db_name, database_conn_args, database_kwarg_dict)

    if udf_load_path is not None and database == DatabaseSystem.DUCKDB:
        # load udf file
        with open(udf_load_path, 'r') as f:
            content = f.read()

        # import udf libraries - this env will be used by duckdb
        db_conn = dbms_wrapper.get_connection()

        # import udf libraries
        import math
        import numpy

        # dummy code to prevent unused import automatic removal
        tmp = math.e
        tmp = numpy.e

        # execute the db_conn.register_function statements
        t_print(f'Exec load udf code from {udf_load_path}', flush=True)

        exec(content, {'math': math, 'numpy': numpy, 'db_conn': db_conn, 'date': datetime.date, 'time': datetime.time,
                       'VARCHAR': duckdb.typing.VARCHAR, 'INTEGER': duckdb.typing.INTEGER,
                       'DOUBLE': duckdb.typing.DOUBLE, 'TIMESTAMP': duckdb.typing.TIMESTAMP, 'DATE': duckdb.typing.DATE,
                       'TIME': duckdb.typing.TIME, 'BOOLEAN': duckdb.typing.BOOLEAN,
                       'HUGEINT': duckdb.typing.HUGEINT, 'FLOAT': duckdb.typing.FLOAT})
        t_print(f'Finished importing udf code from {udf_load_path}', flush=True)

    with open(workload_path) as f:
        content = f.readlines()
    sql_queries = [x.strip() for x in content]

    # rewrite column names to match schema
    sql_queries = [rewrite_query_to_schema(q, db_name) for q in sql_queries]

    # pull up UDFs in the SQL queries
    if pullup_udf or pullup_udf_intermed:
        tmp = []
        for sql_q in sql_queries:
            try:
                rewritten, has_filter_udf = pullup_udf_in_sql(sql_q, intermed_pos=pullup_udf_intermed)
            except Exception as e:
                print(f"Could not pull up UDFs in {sql_q}",flush=True)
                raise e
            if has_filter_udf:
                tmp.append(rewritten)

        sql_queries = tmp

    hint_list = ['' for _ in sql_queries]
    assert hints is None
    # if hints is not None:
    #     if hints == 'random':
    #         hint_list = [gen_optimizer_hint_variant(random.randrange(no_variants)) for _ in sql_queries]
    #
    #     elif hints == 'all':
    #         hint_list = [gen_optimizer_hint_variant(i) for i in range(no_variants) for _ in sql_queries]
    #         sql_queries = [sql_q for i in range(no_variants) for sql_q in sql_queries]
    #
    #     else:
    #         with open(hints) as f:
    #             content = f.readlines()
    #         hint_list = [x.strip() for x in content]
    assert len(hint_list) == len(sql_queries)

    print(f'Vacuuming database {db_name}')
    dbms_wrapper.submit_query('VACUUM ANALYZE;')

    # extract column statistics
    database_stats = dbms_wrapper.collect_db_statistics()

    # remove existing indexes from previous workload runs
    if with_indexes:
        dbms_wrapper.remove_remaining_fk_indexes()

    # check if workload already exists
    query_list = []
    seen_queries = set()
    time_offset = 0
    if os.path.exists(target_path):
        try:
            last_run = load_json(target_path, namespace=False)
            query_list = last_run['query_list']
            if 'total_time_secs' in last_run:
                time_offset = last_run['total_time_secs']
            for q in query_list:
                seen_queries.add(q['sql'])

            if cap_workload is not None:
                print("Checking existing files")
                for q in tqdm(query_list):
                    if check_valid(q, min_runtime=min_runtime, verbose=False, dbms=database, sql_query=q['sql_query']):
                        cap_workload -= 1
                        if cap_workload == 0:
                            print(f"Read existing files already reached sufficient number of queries")
                            return
                print(f"Read existing files and reduced workload cap to {cap_workload}")
        except JSONDecodeError:
            print("Could not read json")

    # set a timeout to make sure long running queries do not delay the entire process
    dbms_wrapper.set_statement_timeout(timeout_sec)

    existing_indexes = dict()

    # extract query plans
    start_t = time.perf_counter()
    valid_queries = 0
    gc.collect()
    for i, sql_query in enumerate(tqdm(sql_queries)):
        if sql_query in seen_queries:
            continue

        if with_indexes:
            modify_indexes(dbms_wrapper, sql_query, existing_indexes, timeout_sec)

        hint = hint_list[i]

        if (pullup_udf or pullup_udf_intermed) and database == DatabaseSystem.DUCKDB:
            dbms_wrapper.submit_query('set disabled_optimizers = \'filter_pushdown\';')

        curr_statistics = dbms_wrapper.run_query_collect_statistics(sql_query, repetitions=repetitions_per_query,
                                                                    prefix=hint)
        curr_statistics.update(sql=sql_query)
        curr_statistics.update(hint=hint)
        query_list.append(curr_statistics)

        run_stats = dict(query_list=query_list,
                         database_stats=database_stats,
                         run_kwargs=run_kwargs,
                         total_time_secs=time_offset + (time.perf_counter() - start_t))

        # save to json
        # write to temporary path and then move
        if len(query_list) % 20 == 0:
            save_workload(run_stats, target_path)

        # check whether sufficient valid queries are available
        if cap_workload is not None:
            try:
                if check_valid(curr_statistics, min_runtime=min_runtime, dbms=database, verbose=True,
                               sql_query=sql_query):
                    valid_queries += 1
            except Exception as e:
                traceback.print_exception(e)
                print(e, flush=True)
                print(curr_statistics, flush=True)

            elapsed_sec = time.perf_counter() - start_t
            remaining_h = 0
            if valid_queries > 0:
                remaining_h = (cap_workload - valid_queries) / valid_queries * elapsed_sec / 3600

            t_print(f"Valid Queries {valid_queries}/{cap_workload} "
                    f"(est. remaining hrs: {remaining_h:.3f}, elapsed secs: {elapsed_sec:.2f})")
            if valid_queries >= cap_workload:
                return

    # Finally remove all indexes again
    if with_indexes:
        dbms_wrapper.remove_remaining_fk_indexes()

    run_stats = dict(query_list=query_list,
                     database_stats=database_stats,
                     run_kwargs=run_kwargs,
                     total_time_secs=time_offset + (time.perf_counter() - start_t),
                     database=str(database))
    save_workload(run_stats, target_path)

    dbms_wrapper.close_conn()

    if udf_drop_path is not None:
        if database == DatabaseSystem.POSTGRES:
            # drop the UDFs
            return_code = subprocess.run(
                f'psql -d ${db_name} -h localhost -p ${database_conn_args["port"]} -U ${database_conn_args["user"]} -c "\i ${udf_drop_path}"',
                shell=True)
            print(f'Drop UDFs exited with code: {return_code}')
        # elif database == DatabaseSystem.DUCKDB:
        #     with open(udf_drop_path, 'r') as f:
        #         content = f.read()
        #         # execute the db_conn.register_function statements
        #         print(f'Exec drop udf code from {udf_drop_path}')
        #         exec(content)

    print(f"Executed workload {workload_path} in {time_offset + time.perf_counter() - start_t:.2f} secs")


def save_workload(run_stats, target_path):
    print(target_path)
    gc.collect()
    try:
        target_temp_path = os.path.join(os.path.dirname(target_path), f'{os.path.basename(target_path)}.temp')
        with open(target_temp_path, 'w') as outfile:
            json.dump(run_stats, outfile)
    except OSError:
        print(f"Could not write to {target_path}")
        target_temp_path = os.path.join(os.path.dirname(target_path), f'{os.path.basename(target_path)}.temp2')
        with open(target_temp_path, 'w') as outfile:
            json.dump(run_stats, outfile)
    # shutil.move(target_temp_path, target_path)
    try:
        Path(target_temp_path).rename(target_path)
    except:
        Path(target_temp_path).replace(target_path)
