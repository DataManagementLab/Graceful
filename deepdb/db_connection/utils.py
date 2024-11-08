from deepdb.db_connection.duckdb import DuckDB
from deepdb.db_connection.postgres import Postgres


def construct_db_connection(dbms: str, db_name: str, duckdb_dir: str = None):
    if dbms == 'duckdb':
        return DuckDB(db_dir=duckdb_dir, db_name=db_name)
    elif dbms == 'postgres':
        return Postgres()
    else:
        raise ValueError(f"Unknown dbms {dbms}")