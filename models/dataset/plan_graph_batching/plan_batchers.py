from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from models.dataset.plan_graph_batching.dd_plan_batching import duckdb_plan_collator

plan_collator_dict = {
    DatabaseSystem.POSTGRES: duckdb_plan_collator,
    DatabaseSystem.DUCKDB: duckdb_plan_collator,
}
