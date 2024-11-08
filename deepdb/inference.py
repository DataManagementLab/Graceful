import functools
from typing import NamedTuple, Any, List, Set, Tuple

from deepdb.datasets.generate_schema import gen_schema
from deepdb.ensemble_compilation.spn_ensemble import read_ensemble
from deepdb.evaluation.augment_cards import convert_query


class FilterCond(NamedTuple):
    table_name: str
    column_name: str
    operator: str
    value: Any


class DeepDBEstimator():
    def __init__(self, ensemble_locations: List[str], db_name: str, scale: int, csv_path: str = None):
        self.ensemble_locations = ensemble_locations
        self.db_name = db_name
        self.scale = scale
        self.csv_path = csv_path

        # load schema
        self.schema = gen_schema(self.db_name, self.csv_path)

        # load the ensemble
        self.spn_ensemble = read_ensemble(self.ensemble_locations, build_reverse_dict=True)

    @functools.lru_cache(maxsize=1000)
    def estimate_card(self, filter_conditions: Tuple[FilterCond], tables: Set[str], join_conditions: List[str] = None):
        """
        Estimate cardinality of a query with filter conditions
        :param filter_conditions: List of filter conditions
        :param tables: List of tables involved in scan / joins and should be considered in the query
        :param join_conditions: List of join conditions - can be empty (in this case they will be derived automatically from the schema - might be wrong when non standard join conditions are used)
        """

        if not isinstance(tables, set):
            tables = set(tables)

        # filter_conditions = [(table_name, column_name, operator, value)]
        filter_conditions = [(fc.table_name, fc.column_name, fc.operator, fc.value) for fc in filter_conditions]

        # convert to query
        q = convert_query(self.schema, tables, filter_conditions, join_conditions, non_inclusive=False)

        # inference the spn and return a cardinality estimate
        _, factors, cardinality_predict, factor_values, evaluate_leaf_nodes, evaluated_sp_nodes, avg_max_depth, num_spns = self.spn_ensemble \
            .cardinality(q, return_factor_values=True, join_keys=self.schema.scaled_join_keys, scale=self.scale,
                         rdc_spn_selection=False,
                         pairwise_rdc_path=None,
                         merge_indicator_exp=True,
                         exploit_overlapping=True, max_variants=1,
                         )

        return cardinality_predict
