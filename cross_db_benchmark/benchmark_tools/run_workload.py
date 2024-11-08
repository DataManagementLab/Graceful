from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.run_workload import run_pg_workload


def run_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path, run_kwargs,
                 repetitions_per_query, timeout_sec, udf_load_path: str = None, udf_drop_path: str = None, hints=None,
                 with_indexes=False, cap_workload=None, min_runtime=100, pullup_udf: bool = False):
    if database in [DatabaseSystem.POSTGRES, DatabaseSystem.DUCKDB]:
        run_pg_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path,
                        run_kwargs, repetitions_per_query, timeout_sec, hints=hints, with_indexes=with_indexes,
                        cap_workload=cap_workload, min_runtime=min_runtime, udf_load_path=udf_load_path,
                        udf_drop_path=udf_drop_path, pullup_udf=pullup_udf)
    else:
        raise NotImplementedError
