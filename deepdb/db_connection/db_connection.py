from deepdb.ensemble_compilation.utils import gen_full_join_query, print_conditions

class DBConnection:

    def get_result(self, sql):
        raise Exception('should be implemented by subclass')


class TrueCardinalityEstimator:
    """Queries the database to return true cardinalities."""

    def __init__(self, schema_graph, db_connection):
        self.schema_graph = schema_graph
        self.db_connection = db_connection

    def true_cardinality(self, query):
        full_join_query = gen_full_join_query(self.schema_graph, query.relationship_set, query.table_set, "JOIN")

        where_cond = print_conditions(query.conditions, seperator='AND')
        if where_cond != "":
            where_cond = "WHERE " + where_cond
        sql_query = full_join_query.format("COUNT(*)", where_cond)
        cardinality = self.db_connection.get_result(sql_query)
        return sql_query, cardinality
