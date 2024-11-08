import multiprocessing
import os

import func_timeout

import duckdb
from deepdb.db_connection.db_connection import DBConnection
from deepdb.utils.utils import t_print


class DuckDB(DBConnection):
    def __init__(self, db_dir: str, db_name: str, version: str = '0.10.1'):
        self.db_dir = db_dir
        self.db_name = db_name

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

    def get_result(self, sql: str, as_df: bool = False, verbose: bool = True):
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
            # assumes the db has already been created
            self.connection = duckdb.connect(
                os.path.join(self.db_dir, f'{self.db}{self.version_str}.db'))
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
