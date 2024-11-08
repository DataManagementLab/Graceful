import unittest

from udf_graph.helper import extract_filter_conds_below_udf_from_plan


class MyTestCase(unittest.TestCase):
    def test_extract_filter_conds_below_udf_from_plan(self):
        plan = {'plan_parameters': {
            'op_name': 'UNGROUPED_AGGREGATE',
            'act_card': 1, 'est_card': 1, 'output_columns': [
                {'aggregation': 'AVG', 'columns': [33, 42], 'udf_name': None, 'udf_output': 'False',
                 'child_ref': False}],
            'act_children_card': 1982960, 'est_children_card': 2369979}, 'children': [{
            'plan_parameters': {
                'op_name': 'HASH_JOIN',
                'act_card': 1982960,
                'est_card': 2369979,
                'join': 'INNER\nOriginCityMarketID = Code',
                'act_children_card': 30486940047,
                'est_children_card': 13103615232},
            'children': [
                {
                    'plan_parameters': {
                        'op_name': 'SEQ_SCAN',
                        'act_card': 10938981,
                        'est_card': 11374666,
                        'table_name': 'On_Time_On_Time_Performance_2016_1',
                        'table': 0,
                        'output_columns': [
                            {
                                'aggregation': 'None',
                                'columns': [
                                    13],
                                'udf_name': None,
                                'udf_output': 'False',
                                'child_ref': False},
                            {
                                'aggregation': 'None',
                                'columns': [
                                    33],
                                'udf_name': None,
                                'udf_output': 'False',
                                'child_ref': False},
                            {
                                'aggregation': 'None',
                                'columns': [
                                    42],
                                'udf_name': None,
                                'udf_output': 'False',
                                'child_ref': False}],
                        'act_children_card': 1,
                        'est_children_card': 1},
                    'children': []},
                {
                    'plan_parameters': {
                        'op_name': 'FILTER',
                        'act_card': 2787,
                        'est_card': 1152,
                        'filter_columns': {
                            'text': 'func_24 >= 32670',
                            'children': [],
                            'column': None,
                            'operator': '>=',
                            'literal': '32670',
                            'literal_feature': 0,
                            'udf_name': 'func_24'},
                        'udf_params': [
                            103],
                        'udf_table': 11,
                        'act_children_card': 5079,
                        'est_children_card': 1152},
                    'children': [
                        {
                            'plan_parameters': {
                                'op_name': 'SEQ_SCAN',
                                'act_card': 5079,
                                'est_card': 1152,
                                'table_name': 'L_CITY_MARKET_ID',
                                'table': 11,
                                'output_columns': [
                                    {
                                        'aggregation': 'None',
                                        'columns': [
                                            103],
                                        'udf_name': None,
                                        'udf_output': 'False',
                                        'child_ref': False}],
                                'filter_columns': {
                                    'text': ' AND ',
                                    'children': [
                                        {
                                            'text': ' Code>=30070',
                                            'children': [],
                                            'column': 103,
                                            'operator': '>=',
                                            'literal': '30070',
                                            'literal_feature': 0,
                                            'udf_name': None},
                                        {
                                            'text': 'Code<=35992',
                                            'children': [],
                                            'column': 103,
                                            'operator': '<=',
                                            'literal': '35992',
                                            'literal_feature': 0,
                                            'udf_name': None},
                                        {
                                            'text': 'Code IS NOT NULL ',
                                            'children': [],
                                            'column': 103,
                                            'operator': 'IS NOT NULL',
                                            'literal': '',
                                            'literal_feature': 0,
                                            'udf_name': None}],
                                    'column': None,
                                    'operator': 'AND',
                                    'literal': None,
                                    'literal_feature': None,
                                    'udf_name': None},
                                'act_children_card': 1,
                                'est_children_card': 1},
                            'children': []}]}]}],
            'udf': {'udf_name': 'func_24', 'udf_math_lib_imported': False, 'udf_numpy_lib_imported': True,
                    'udf_num_math_calls': 0, 'udf_num_np_calls': 2, 'udf_pos_in_query': 'filter'},
            'query': 'SELECT AVG("On_Time_On_Time_Performance_2016_1"."DepDel15" + "On_Time_On_Time_Performance_2016_1"."ArrDelay") as agg_0 FROM "L_CITY_MARKET_ID" JOIN "On_Time_On_Time_Performance_2016_1" ON "L_CITY_MARKET_ID"."Code" = "On_Time_On_Time_Performance_2016_1"."OriginCityMarketID"  WHERE func_24("L_CITY_MARKET_ID"."Code") >= 32670;',
            'plan_runtime_ms': 271.22850000000005, 'num_tables': 2, 'num_filters': 3}

        where_str = extract_filter_conds_below_udf_from_plan(plan)
        self.assertEqual(where_str, [
            'L_CITY_MARKET_ID.Code>=30070 AND L_CITY_MARKET_ID.Code<=35992 AND L_CITY_MARKET_ID.Code IS NOT NULL'])


if __name__ == '__main__':
    unittest.main()
