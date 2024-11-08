import gc
import json
import multiprocessing
import os
import traceback

import duckdb
import func_timeout

from cross_db_benchmark.benchmark_tools.database import DatabaseConnection
from cross_db_benchmark.benchmark_tools.utils import load_schema_json
from utils.utils import t_print


class DuckdbDatabaseConnection(DatabaseConnection):
    def __init__(self, version: str = '0.10.1', **kwargs, ):
        super().__init__(**kwargs)
        self.connection = None
        self.timeout_sec = None
        self.profiling_enabled = False
        self.version = version

        if self.version == '0.9.2':
            self.version_str = ''
        elif self.version == '0.10.1':
            self.version_str = '_10_1'
        else:
            raise ValueError(f"Unknown version {self.version}")

        print(f'Using version {self.version} / {self.version_str} of DuckDB')

    def load_database(self, dataset, data_dir, force=False):
        # use zs initialization repo for that
        raise NotImplementedError()

    def replicate_tuples(self, dataset, data_dir, no_prev_replications, vac_analyze=True):
        # use zs initialization repo for that
        raise NotImplementedError()

    def check_if_database_exists(self):
        db_dir = self.database_kwargs['dir']
        path = os.path.join(db_dir, f'{self.db_name}{self.version_str}.db')
        return os.path.exists(path)

    def remove_remaining_fk_indexes(self):
        idx = self.get_result("SELECT index_name FROM duckdb_indexes;", as_df=True).values[:, 0]

        for idx_name in idx:
            self.drop_index(idx_name)

    def drop_index(self, index_name):
        print(f"Dropping index {index_name}")
        self.submit_query(f'DROP INDEX "{index_name}";')

    def create_index(self, table, column):
        index_name = f"zero_shot_{table}_{column}"
        self.submit_query(f'CREATE INDEX "{index_name}" ON "{table}" ("{column}");')
        return index_name

    def create_db(self):
        # use zs initialization repo for that
        raise NotImplementedError()

    def test_join_conditions(self, dataset):
        schema = load_schema_json(dataset)

        for table_left, cols_left, table_right, cols_right in schema.relationships:
            if not (isinstance(cols_left, tuple) or isinstance(cols_left, list)):
                cols_left = [cols_left]
                cols_right = [cols_right]

            join_conds = ' AND '.join([f'"{table_left}"."{c_left}" = "{table_right}"."{c_right}"'
                                       for c_left, c_right in zip(cols_left, cols_right)])

            res = self.get_result(f"SELECT COUNT(*) FROM \"{table_left}\" JOIN \"{table_right}\" ON {join_conds};",
                                  as_df=True)
            card = res.values[0][0]
            print(f"{join_conds}: {card} join tuples")
            if not card > 1:
                print("WARNING: low cardinality. Check join condition")

    def set_statement_timeout(self, timeout_sec, verbose=True):
        # timeout needs to be passed at exec time
        self.timeout_sec = timeout_sec

    def run_query_collect_statistics(self, sql, repetitions=3, prefix="", explain_only=False):
        analyze_plans = None
        verbose_plan = None
        timeout = False

        if not self.profiling_enabled:
            # enable json output for profiling information
            self.submit_query("PRAGMA enable_profiling = 'json';")
            self.profiling_enabled = True

        try:
            verbose_plan = self.get_result(f"{prefix}EXPLAIN {sql}", as_df=False, verbose=False)

            analyze_plans = []
            if not explain_only:
                import math
                import numpy
                import datetime
                import time
                for i in range(repetitions):
                    statement = f"{prefix}EXPLAIN ANALYZE {sql}"
                    curr_analyze_plan = self.get_result(statement, as_df=False, verbose=False)
                    curr_analyze_plan_json = json.loads(curr_analyze_plan[0][1])
                    analyze_plans.append(curr_analyze_plan_json)
        except func_timeout.exceptions.FunctionTimedOut:
            gc.collect()
            t_print(f"Timeout ({self.timeout_sec}s): {sql}", flush=True)
            timeout = True
        except duckdb.InvalidInputException as e:
            t_print(f"Invalid Input Exception {e}: {sql}", flush=True)
        except duckdb.BinderException as e:
            t_print(f"Binder error: {sql}", flush=True)
        except Exception as e:
            print(e, flush=True)
            traceback.print_exc()
            t_print(f"Skipping query {sql} due to an error {e}", flush=True)

        return dict(analyze_plans=analyze_plans, verbose_plan=verbose_plan, timeout=timeout)

    def collect_db_statistics(self):
        # column stats
        stats_query = """
        SELECT table_name, column_name, data_type FROM duckdb_columns;
        """
        column_stats_names = ['table_name', 'column_name', 'data_type']
        column_stats_rows = self.get_result(stats_query, as_df=True).values
        column_stats = self.transform_dicts(column_stats_names, column_stats_rows)

        # table stats
        stats_query = """
        select table_name, estimated_size from duckdb_tables;
         """
        table_stats_names = ['table_name', 'estimated_size']
        table_stats_rows = self.get_result(stats_query, as_df=True).values
        table_stats = self.transform_dicts(table_stats_names, table_stats_rows)

        return dict(column_stats=column_stats, table_stats=table_stats)

    def get_result(self, sql: str, as_df: bool, verbose: bool = True):
        conn = None
        try:
            conn = self.get_connection()

            if self.timeout_sec is not None and self.timeout_sec != 0:
                # Create a Process
                p = multiprocessing.Process(target=run_query, args=(conn, sql))
                # Start the process
                p.start()
                # Wait for the process to finish or kill it after the timeout
                p.join(timeout=self.timeout_sec)
                if p.is_alive():
                    p.terminate()
                    raise func_timeout.exceptions.FunctionTimedOut()
                else:
                    records = conn.sql(sql)
            else:
                records = conn.sql(sql)

            if as_df:
                return records.fetchdf()
            else:
                res = records.fetchall()
                return res
        except func_timeout.exceptions.FunctionTimedOut as e:
            if verbose:
                t_print(f'query {sql} timed out', flush=True)
                t_print(f'query interrupted', flush=True)
            try:
                duckdb.interrupt(connection=conn)
            except Exception as interrupt_exc:
                t_print(f'Error while interrupting query {sql}: {interrupt_exc}', flush=True)
            raise e
        except Exception as e:
            if verbose:
                t_print(f"Error while executing {sql}", flush=True)
            raise e

    def get_connection(self):
        if self.connection is None:
            assert os.path.exists(
                self.database_kwargs['dir']), f"Database directory {self.database_kwargs['dir']} does not exist"
            db_path = os.path.join(self.database_kwargs['dir'], f'{self.db_name}{self.version_str}.db')
            assert os.path.exists(db_path), f"Database {db_path} does not exist"

            # assumes the db has already been created
            self.connection = duckdb.connect(
                db_path)
            self.profiling_enabled = False
        return self.connection

    def submit_query(self, sql):
        conn = self.get_connection()
        if self.timeout_sec is not None and self.timeout_sec != 0:
            func_timeout.func_timeout(timeout=self.timeout_sec, func=conn.execute, args=(sql,))
        else:
            conn.execute(sql)

    def close_conn(self):
        print(f'Closing connection to {self.db_name}{self.version_str}')
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            self.profiling_enabled = False


def run_query(conn, sql):
    res = conn.sql(sql)
    return res
