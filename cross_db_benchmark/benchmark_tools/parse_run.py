import json
import os
from enum import Enum
from typing import Dict, List

from deepdb.inference import DeepDBEstimator

from cross_db_benchmark.benchmark_tools.annotate_cards_for_udf_sels import annotate_udf_info_to_plan
from cross_db_benchmark.benchmark_tools.augment_deepdb_card import annotate_deepdb_card
from cross_db_benchmark.benchmark_tools.augment_wanderjon_card import annotate_wanderjoin_card
from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.combine_plans import combine_traces, combine_dd_traces
from cross_db_benchmark.benchmark_tools.postgres.parse_dd_plan import parse_dd_plans
from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_plans
from cross_db_benchmark.benchmark_tools.utils import load_json, load_schema_json


def dumper(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, Enum):
        return str(obj)
    try:
        return obj.toJSON()
    except:
        try:
            return obj.to_dict()
        except:
            try:
                return obj.__dict__
            except Exception as e:
                print(f'Obj: {type(obj)} / {obj}')
                raise e


def parse_run(source_paths, target_path, db_name: str, deepdb_ensemble_locations: List[str],
              deepdb_dataset_scale_factor: int,
              database, duckdb_kwargs: Dict = None, pg_kwargs: Dict = None, min_query_ms=100,
              max_query_ms=30000, parse_baseline=False, cap_queries=None, parse_join_conds=False,
              include_zero_card=False, explain_only=False, udf_code_location: str = None, skip_dump: bool = False,
              skip_wj: bool = False, skip_deepdb: bool = False, keep_existing: bool = False, prune_plans: bool = False):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if database == DatabaseSystem.POSTGRES:
        parse_func = parse_plans
        comb_func = combine_traces
    elif database == DatabaseSystem.DUCKDB:
        parse_func = parse_dd_plans
        comb_func = combine_dd_traces
    else:
        raise NotImplementedError(f"Database {database} not yet supported.")

    if not isinstance(source_paths, list):
        source_paths = [source_paths]

    assert all([os.path.exists(p) for p in source_paths])

    if udf_code_location is not None:
        udf_path = os.path.join(udf_code_location, db_name, 'sql_scripts', 'udfs.json')
        assert os.path.exists(udf_path), f"UDF code location {udf_path} does not exist."
        with open(udf_path) as f:
            udf_code_dict = json.load(f)
    else:
        udf_code_dict = None

    run_stats = [load_json(p) for p in source_paths]
    run_stats = comb_func(run_stats)

    if keep_existing:
        with open(target_path, 'r') as f:
            parsed_runs = json.load(f)
        stats = dict()
    else:
        parsed_runs, stats = parse_func(run_stats, min_runtime_ms=min_query_ms, max_runtime_ms=max_query_ms,
                                        parse_baseline=parse_baseline, cap_queries=cap_queries,
                                        parse_join_conds=parse_join_conds,
                                        include_zero_card=include_zero_card, explain_only=explain_only,
                                        udf_code_dict=udf_code_dict, prune_plans=prune_plans)

    # load schema relationships
    schema = load_schema_json(dataset=db_name)
    schema_relationships = schema.relationships

    # annotate deepdb cardinalities
    if not skip_deepdb:
        print("Annotating deepdb cardinalities")

        # load deepdb instance
        deepdb_estimator = DeepDBEstimator(ensemble_locations=deepdb_ensemble_locations, db_name=db_name,
                                           scale=deepdb_dataset_scale_factor)

        annotate_deepdb_card(parsed_runs=parsed_runs, deepdb_estimator=deepdb_estimator,
                             schema_relationships=schema_relationships)

    # annotate wanderjoin cardinalities
    if not skip_wj:
        print("Annotating wanderjoin cardinalities")
        annotate_wanderjoin_card(parsed_runs=parsed_runs, duckdb_kwargs=duckdb_kwargs, pg_kwargs=pg_kwargs,
                                 schema_relationships=schema_relationships)

    # annotate different card ests for assumed UDF filter selectivities (i.e. add est_card_10, est_card_30, ... , est_card_90 features)
    annotate_udf_info_to_plan(parsed_runs=parsed_runs)

    if not skip_dump:
        with open(target_path, 'w') as outfile:
            if database == DatabaseSystem.POSTGRES:
                json.dump(parsed_runs, outfile, default=dumper)
            elif database == DatabaseSystem.DUCKDB:
                json.dump(parsed_runs, outfile, default=dumper)
            else:
                raise NotImplementedError(f"Database {database} not yet supported.")
    else:
        print(f"Skipping dump of parsed plans to {target_path}")
    return len(parsed_runs['parsed_plans']), stats
