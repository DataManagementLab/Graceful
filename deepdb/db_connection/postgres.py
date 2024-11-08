try:
    import psycopg2
except ImportError:
    pass
import pandas as pd
from deepdb.db_connection.db_connection import DBConnection


class Postgres(DBConnection):

    def __init__(self, db_user="postgres", db_password="postgres", db_host="localhost", db_port="5432", db="shopdb"):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db = db

    def _vacuum(self):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        old_isolation_level = connection.isolation_level
        connection.set_isolation_level(0)
        query = "VACUUM"
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
        connection.set_isolation_level(old_isolation_level)

    def _get_dataframe(self, sql):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        return pd.read_sql(sql, connection)

    def _submit_query(self, sql):
        """Submits query and ignores result."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()
        cursor.execute(sql)
        connection.commit()

    def get_result(self, sql):
        """Fetches exactly one row of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()

        cursor.execute(sql)
        record = cursor.fetchone()
        result = record[0]

        if connection:
            cursor.close()
            connection.close()

        return result

    def _get_result_set(self, sql, return_columns=False):
        """Fetches all rows of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()

        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        if connection:
            cursor.close()
            connection.close()

        if return_columns:
            return rows, columns

        return rows
