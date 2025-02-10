import json
import unittest
from types import SimpleNamespace

from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator
from cross_db_benchmark.benchmark_tools.dbms.parse_dd_plan import parse_plan_recursively, parse_output_columns, \
    remove_cast_nesting
from cross_db_benchmark.benchmark_tools.dbms.parse_filter import PredicateNode

table_id_mapping = {'oseba': 0, 'nesreca': 1, 'part': 2, 'regular_season_detailed_results': 3, 'team_name': 4,
                    'teams': 5, 'orders': 6, 'account': 7, 'loan': 8, 'district': 9, 'lineitem': 10,
                    'nation': 11, 'region': 12, 'supplier': 13, 'partsupp': 14, 'movie_keyword': 15,
                    'keyword': 16}
column_id_mapping = {('oseba', 'id_nesreca'): 0, ('oseba', 'povzrocitelj_ali_udelezenec'): 1,
                     ('oseba', 'vrsta_udelezenca'): 2, ('oseba', 'starost'): 2, ('oseba', 'spol'): 3,
                     ('oseba', 'upravna_enota'): 4, ('oseba', 'drzavljanstvo'): 5, ('oseba', 'poskodba'): 6,
                     ('part', 'p_size'): 7, ('part', 'p_mfgr'): 8, ('part', 'p_brand'): 9,
                     ('part', 'p_type'): 10, ('part', 'p_partkey'): 11, ('part', 'p_name'): 12,
                     ('oseba', 'varnostni_pas_ali_celada'): 13, ('oseba', 'vozniski_staz_LL'): 14,
                     ('oseba', 'vozniski_staz_MM'): 15, ('oseba', 'alkotest'): 16,
                     ('oseba', 'strokovni_pregled'): 17, ('oseba', 'starost_d'): 18,
                     ('oseba', 'vozniski_staz_d'): 19, ('oseba', 'alkotest_d'): 20,
                     ('oseba', 'strokovni_pregled_d'): 21, ('nesreca', 'id_nesreca'): 22, ('nesreca', 'x'): 23,
                     ('regular_season_detailed_results', 'lfta'): 24,
                     ('regular_season_detailed_results', 'lteam'): 25,
                     ('regular_season_detailed_results', 'lstl'): 26,
                     ('regular_season_detailed_results', 'lor'): 27,
                     ('regular_season_detailed_results', 'wteam'): 28,
                     ('regular_season_detailed_results', 'wfga3'): 29,
                     ('regular_season_detailed_results', 'lpf'): 30,
                     ('regular_season_detailed_results', 'lscore'): 31,
                     ('regular_season_detailed_results', 'wfta'): 32,
                     ('team_name', 'team_id'): 33, ('team_name', 'team_name'): 34, ('teams', 'team_id'): 35,
                     ('teams', 'team_name'): 36, ('nesreca', 'y'): 37, ('nesreca', 'y_wgs84'): 38,
                     ('nesreca', 'klas_nesreca'): 39, ('nesreca', 'vzrok_nesreca'): 40,
                     ('nesreca', 'tekst_cesta_ali_naselje'): 41, ('nesreca', 'stanje_promet'): 42,
                     ('nesreca', 'vreme_nesreca'): 43, ('nesreca', 'kategorija_cesta'): 44,
                     ('nesreca', 'oznaka_cesta_ali_naselje'): 45, ('nesreca', 'opis_prizorisce'): 46,
                     ('nesreca', 'stanje_povrsina_vozisce'): 47, ('nesreca', 'cas_nesreca'): 48,
                     ('nesreca', 'tip_nesreca'): 49, ('nesreca', 'naselje_ali_izven'): 50,
                     ('nesreca', 'upravna_enota'): 51, ('nesreca', 'tekst_odsek_ali_ulica'): 52,
                     ('nesreca', 'oznaka_odsek_ali_ulica'): 53, ('orders', 'account_id'): 54,
                     ('account', 'account_id'): 55, ('loan', 'account_id'): 56, ('loan', 'duration'): 57,
                     ('account', 'district_id'): 58, ('orders', 'order_id'): 59, ('orders', 'account_to'): 60,
                     ('orders', 'bank_to'): 61, ('orders', 'amount'): 62, ('district', 'district_id'): 63,
                     ('nation', 'n_regionkey'): 64, ('region', 'r_regionkey'): 65,
                     ('nation', 'n_nationkey'): 66, ('supplier', 's_nationkey'): 67,
                     ('supplier', 's_suppkey'): 68, ('lineitem', 'l_suppkey'): 69,
                     ('lineitem', 'l_orderkey'): 70, ('lineitem', 'l_quantity'): 71,
                     ('lineitem', 'l_extendedprice'): 72, ('lineitem', 'l_discount'): 73,
                     ('lineitem', 'l_tax'): 74, ('lineitem', 'l_returnflag'): 75,
                     ('lineitem', 'l_linestatus'): 76, ('lineitem', 'l_shipdate'): 77,
                     ('lineitem', 'l_commitdate'): 78, ('lineitem', 'l_receiptdate'): 79,
                     ('lineitem', 'l_shipinstruct'): 80, ('lineitem', 'l_shipmode'): 81,
                     ('lineitem', 'l_comment'): 82, ('supplier', 's_suppkey'): 83, ('supplier', 's_name'): 84,
                     ('supplier', 's_address'): 85, ('supplier', 's_phone'): 86, ('supplier', 's_acctbal'): 87,
                     ('supplier', 's_comment'): 88, ('lineitem', 'l_partkey'): 89,
                     ('partsupp', 'ps_availqty'): 90, ('partsupp', 'ps_supplycost'): 91,
                     ('partsupp', 'ps_partkey'): 92, ('partsupp', 'ps_suppkey'): 93,
                     ('movie_keyword', 'keyword_id'): 94, ('movie_keyword', 'id'): 95,
                     ('movie_keyword', 'keyword'): 96, ('movie_keyword', 'movie_id'): 97, ('keyword', 'id'): 98,
                     ('keyword', 'keyword'): 99, ('keyword', 'phonetic_code'): 100}


class MyTestCase(unittest.TestCase):
    def test_parse_plan_recursively(self):
        plan = {
            "name": "Query",
            "result": 0.201727,
            "timing": 0.201739,
            "cardinality": 0,
            "extra-info": "explain analyze select * from oseba, nesreca where oseba.id_nesreca = nesreca.id_nesreca and nesreca.x > 10 limit 10;",
            "timings": [

            ],
            "children": [
                {
                    "name": "EXPLAIN_ANALYZE",
                    "timing": 0.000000,
                    "cardinality": 0,
                    "extra_info": "",
                    "timings": [
                    ],
                    "children": [
                        {
                            "name": "STREAMING_LIMIT",
                            "timing": 0.000008,
                            "cardinality": 10,
                            "extra_info": "",
                            "timings": [
                            ],
                            "children": [
                                {
                                    "name": "PROJECTION",
                                    "timing": 0.000101,
                                    "cardinality": 10497,
                                    "extra_info": "id_nesreca\npovzrocitelj_ali_udelezenec\nstarost\nspol\nupravna_enota\ndrzavljanstvo\nposkodba\nvrsta_udelezenca\nvarnostni_pas_ali_celada\nvozniski_staz_LL\nvozniski_staz_MM\nalkotest\nstrokovni_pregled\nstarost_d\nvozniski_staz_d\nalkotest_d\nstrokovni_pregled_d\nid_nesreca\n",
                                    "timings": [
                                    ],
                                    "children": [
                                        {
                                            "name": "HASH_JOIN",
                                            "timing": 0.445044,
                                            "cardinality": 10497,
                                            "extra_info": "INNER\nid_nesreca = id_nesreca\n\n[INFOSEPARATOR]\nEC: 159439\n",
                                            "timings": [
                                            ],
                                            "children": [
                                                {
                                                    "name": "SEQ_SCAN ",
                                                    "timing": 0.038440,
                                                    "cardinality": 14336,
                                                    "extra_info": "oseba\n[INFOSEPARATOR]\nid_nesreca\npovzrocitelj_ali_udelezenec\nstarost\nspol\nupravna_enota\ndrzavljanstvo\nposkodba\nvrsta_udelezenca\nvarnostni_pas_ali_celada\nvozniski_staz_LL\nvozniski_staz_MM\nalkotest\nstrokovni_pregled\nstarost_d\nvozniski_staz_d\nalkotest_d\nstrokovni_pregled_d\n[INFOSEPARATOR]\nEC: 750000",
                                                    "timings": [
                                                    ],
                                                    "children": [
                                                    ]
                                                }
                                                ,
                                                {
                                                    "name": "SEQ_SCAN ",
                                                    "timing": 0.180227,
                                                    "cardinality": 416090,
                                                    "extra_info": "nesreca\n[INFOSEPARATOR]\nid_nesreca\nx\n[INFOSEPARATOR]\nFilters: x>=10 AND x IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 101798",
                                                    "timings": [
                                                    ],
                                                    "children": [
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        plan2 = {
            "name": "SEQ_SCAN ",
            "timing": 0.003763,
            "cardinality": 200000,
            "extra_info": "part\n[INFOSEPARATOR]\np_partkey\np_size\np_mfgr\np_brand\np_type\n[INFOSEPARATOR]\nEC: 130000",
            "timings": [],
            "children": []
        }

        plan = parse_plan_recursively(json.loads(json.dumps(plan), object_hook=lambda d: SimpleNamespace(**d)),
                                      include_zero_card=True, table_id_mapping=table_id_mapping,
                                      column_id_mapping=column_id_mapping)
        self.assertEqual(plan['children'][0]['children'][0]['children'][1]['plan_parameters']['op_name'], 'SEQ_SCAN')

        plan = parse_plan_recursively(json.loads(json.dumps(plan2), object_hook=lambda d: SimpleNamespace(**d)),
                                      include_zero_card=True, table_id_mapping=table_id_mapping,
                                      column_id_mapping=column_id_mapping)
        expected_plan = {'children': [],
                         'plan_parameters': {'act_card': 200000,
                                             'act_children_card': 1,
                                             'est_card': 130000,
                                             'est_children_card': 1,
                                             'op_name': 'SEQ_SCAN',
                                             'output_columns': [
                                                 {'aggregation': 'None', 'columns': [11], 'udf_name': None,
                                                  'udf_output': 'False', 'child_ref': False},
                                                 {'aggregation': 'None', 'columns': [7], 'udf_name': None,
                                                  'udf_output': 'False', 'child_ref': False},
                                                 {'aggregation': 'None', 'columns': [8], 'udf_name': None,
                                                  'udf_output': 'False', 'child_ref': False},
                                                 {'aggregation': 'None', 'columns': [9], 'udf_name': None,
                                                  'udf_output': 'False', 'child_ref': False},
                                                 {'aggregation': 'None', 'columns': [10], 'udf_name': None,
                                                  'udf_output': 'False', 'child_ref': False}],
                                             'table': 2,
                                             'table_name': 'part'}}
        self.assertEqual(expected_plan, plan)

        raw_plan3 = {
            "name": "Query",
            "result": 5.616614,
            "timing": 5.616695,
            "cardinality": 0,
            "extra-info": "EXPLAIN ANALYZE SELECT AVG(\"regular_season_detailed_results\".\"lfta\" + \"regular_season_detailed_results\".\"lteam\") as agg_0, SUM(\"regular_season_detailed_results\".\"lstl\" + \"regular_season_detailed_results\".\"lor\") as agg_1 FROM \"regular_season_detailed_results\" JOIN \"teams\" ON \"regular_season_detailed_results\".\"wteam\" = \"teams\".\"team_id\"  WHERE func_14(\"regular_season_detailed_results\".\"lscore\",\"regular_season_detailed_results\".\"wfga3\",\"regular_season_detailed_results\".\"lpf\") != 41 AND \"teams\".\"team_id\" <= 61717 AND \"teams\".\"team_name\" != 'Charleston So' AND \"regular_season_detailed_results\".\"wteam\" <= 25856 AND \"regular_season_detailed_results\".\"wfta\" >= 24;",
            "timings": [],
            "children": [
                {
                    "name": "EXPLAIN_ANALYZE",
                    "timing": 0,
                    "cardinality": 0,
                    "extra_info": "",
                    "timings": [],
                    "children": [
                        {
                            "name": "UNGROUPED_AGGREGATE",
                            "timing": 0.005341,
                            "cardinality": 1,
                            "extra_info": "avg(#0)\nsum(#1)\nsum_no_overflow(#1)",
                            "timings": [],
                            "children": [
                                {
                                    "name": "PROJECTION",
                                    "timing": 0.008041,
                                    "cardinality": 509371,
                                    "extra_info": "(lfta + lteam)\n(lstl + lor)\n",
                                    "timings": [],
                                    "children": [
                                        {
                                            "name": "HASH_JOIN",
                                            "timing": 0.012103,
                                            "cardinality": 509371,
                                            "extra_info": "INNER\nwteam = team_id\n\n[INFOSEPARATOR]\nEC: 4841\n",
                                            "timings": [],
                                            "children": [
                                                {
                                                    "name": "FILTER",
                                                    "timing": 44.39921,
                                                    "cardinality": 510153,
                                                    "extra_info": "(func_14(CAST(lscore AS BIGINT), CAST(wfga3 AS BIGINT), CAST(lpf AS BIGINT)) != 41)\n[INFOSEPARATOR]\nEC: 121036",
                                                    "timings": [],
                                                    "children": [
                                                        {
                                                            "name": "SEQ_SCAN ",
                                                            "timing": 0.07803,
                                                            "cardinality": 510170,
                                                            "extra_info": "regular_season_detailed_results\n[INFOSEPARATOR]\nwteam\nlscore\nwfga3\nlpf\nlfta\nlteam\nlstl\nlor\n[INFOSEPARATOR]\nFilters: wfta>=24 AND wfta IS NOT NULL\nwteam<=25856 AND wteam IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 121036",
                                                            "timings": [],
                                                            "children": []
                                                        }
                                                    ]
                                                },
                                                {
                                                    "name": "FILTER",
                                                    "timing": 0.0001,
                                                    "cardinality": 6171,
                                                    "extra_info": "(team_name != 'Charleston So')\n[INFOSEPARATOR]\nEC: 728",
                                                    "timings": [],
                                                    "children": [
                                                        {
                                                            "name": "SEQ_SCAN ",
                                                            "timing": 0.000164,
                                                            "cardinality": 6188,
                                                            "extra_info": "teams\n[INFOSEPARATOR]\nteam_id\nteam_name\n[INFOSEPARATOR]\nFilters: team_id<=25856 AND team_id IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 728",
                                                            "timings": [],
                                                            "children": []
                                                        }
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        expected_plan3 = {
            'plan_parameters': {
                'act_card': 1,
                'act_children_card': 509371,
                'est_card': -1,
                'est_children_card': 4841,
                'op_name': 'UNGROUPED_AGGREGATE',
                'output_columns': [
                    {'aggregation': 'AVG', 'columns': [24, 25], 'child_ref': False,
                     'udf_name': None, 'udf_output': 'False'},
                    {'aggregation': 'SUM', 'columns': [26, 27], 'child_ref': False,
                     'udf_name': None, 'udf_output': 'False'},
                    {'aggregation': 'SUM', 'columns': [26, 27], 'child_ref': False,
                     'udf_name': None, 'udf_output': 'False'}
                ],
            },
            'children': [
                {'plan_parameters': {
                    'act_card': 509371, 'est_card': 4841, 'op_name': 'HASH_JOIN',
                    'join': 'INNER\nwteam = team_id',
                    'act_children_card': 3148154163, 'est_children_card': 88114208},
                    'children': [
                        {'plan_parameters': {
                            'act_card': 510153, 'est_card': 121036, 'op_name': 'FILTER',
                            'act_children_card': 510170, 'est_children_card': 121036,
                            'filter_columns':
                                PredicateNode(operator=Operator.NEQ, literal='41', column=None,
                                              children=[], text=None, literal_feature=0, udf_name='func_14', ),
                            'udf_params': [31, 29, 30],
                            'udf_table': 3
                        },
                            'children': [
                                {'plan_parameters': {
                                    'act_card': 510170, 'est_card': 121036,
                                    'act_children_card': 1, 'est_children_card': 1,
                                    'op_name': 'SEQ_SCAN',
                                    'table': 3,
                                    'table_name': 'regular_season_detailed_results',
                                    'filter_columns': PredicateNode(
                                        text=' AND ',
                                        operator=LogicalOperator.AND,
                                        children=[
                                            PredicateNode(operator=Operator.GEQ, literal='24', column=32,
                                                          children=[], text=None, literal_feature=0),
                                            PredicateNode(operator=Operator.IS_NOT_NULL, column=32, children=[],
                                                          text=None, literal_feature=0, literal=''),
                                            PredicateNode(operator=Operator.LEQ, literal='25856', column=28,
                                                          children=[], text=None, literal_feature=0),
                                            PredicateNode(operator=Operator.IS_NOT_NULL, column=28, children=[],
                                                          text=None, literal_feature=0, literal=''),
                                        ]
                                    ),
                                    'output_columns': [
                                        {'aggregation': 'None', 'columns': [28], 'udf_name': None, 'child_ref': False,
                                         'udf_output': 'False'},
                                        {'aggregation': 'None', 'columns': [31], 'udf_name': None,
                                         'child_ref': False, 'udf_output': 'False'},
                                        {'aggregation': 'None', 'columns': [29], 'udf_name': None,
                                         'child_ref': False, 'udf_output': 'False'},
                                        {'aggregation': 'None', 'columns': [30], 'udf_name': None,
                                         'child_ref': False, 'udf_output': 'False'},
                                        {'aggregation': 'None', 'columns': [24], 'udf_name': None, 'child_ref': False,
                                         'udf_output': 'False'},
                                        {'aggregation': 'None', 'columns': [25], 'udf_name': None,
                                         'child_ref': False, 'udf_output': 'False'},
                                        {'aggregation': 'None', 'columns': [26], 'udf_name': None,
                                         'child_ref': False, 'udf_output': 'False'},
                                        {'aggregation': 'None', 'columns': [27], 'udf_name': None,
                                         'child_ref': False, 'udf_output': 'False'},
                                    ], },
                                    'children': []}
                            ]},
                        {'plan_parameters': {'act_card': 6171, 'est_card': 728, 'op_name': 'FILTER',
                                             'act_children_card': 6188, 'est_children_card': 728,
                                             'filter_columns':
                                                 PredicateNode(operator=Operator.NEQ, literal="'Charleston So'",
                                                               column=36, literal_feature=0),

                                             }, 'children': [
                            {'plan_parameters': {'act_card': 6188, 'est_card': 728,
                                                 'act_children_card': 1, 'est_children_card': 1,
                                                 'op_name': 'SEQ_SCAN', 'table': 5,
                                                 'filter_columns': PredicateNode(
                                                     operator=LogicalOperator.AND,
                                                     children=[
                                                         PredicateNode(operator=Operator.LEQ, literal='25856',
                                                                       column=35, literal_feature=0),
                                                         PredicateNode(operator=Operator.IS_NOT_NULL,
                                                                       column=35, literal_feature=0,
                                                                       literal=''),
                                                     ]
                                                 ),
                                                 'output_columns': [
                                                     {'aggregation': 'None', 'columns': [35],
                                                      'udf_name': None, 'child_ref': False, 'udf_output': 'False'},
                                                     {'aggregation': 'None', 'columns': [36],
                                                      'udf_name': None, 'child_ref': False, 'udf_output': 'False'}
                                                 ],
                                                 'table_name': 'teams'},
                             'children': []}
                        ]}

                    ]

                }
            ]}
        parsed_plan3 = parse_plan_recursively(
            json.loads(json.dumps(raw_plan3), object_hook=lambda d: SimpleNamespace(**d)),
            include_zero_card=True, table_id_mapping=table_id_mapping,
            column_id_mapping=column_id_mapping)
        self.maxDiff = None
        self.assertEqual(expected_plan3, parsed_plan3)

        raw_plan4 = {
            "name": "Query", "result": 24.812415, "timing": 24.812444, "cardinality": 0,
            "extra-info": "EXPLAIN ANALYZE SELECT func_0(\"y\",\"x\",\"y_wgs84\",\"upravna_enota\",\"tekst_odsek_ali_ulica\",\"oznaka_odsek_ali_ulica\",\"klas_nesreca\",\"vzrok_nesreca\",\"tekst_cesta_ali_naselje\",\"stanje_promet\",\"vreme_nesreca\",\"kategorija_cesta\",\"oznaka_cesta_ali_naselje\",\"opis_prizorisce\",\"stanje_povrsina_vozisce\",\"cas_nesreca\",\"tip_nesreca\",\"naselje_ali_izven\",\"id_nesreca\") FROM \"nesreca\";",
            "timings": [], "children": [
                {"name": "EXPLAIN_ANALYZE", "timing": 0.000254, "cardinality": 0, "extra_info": "", "timings": [],
                 "children": [
                     {"name": "PROJECTION", "timing": 95.238563, "cardinality": 416072,
                      "extra_info": "func_0(CAST(y AS BIGINT), CAST(x AS BIGINT), y_wgs84)\n",
                      "timings": [],
                      "children": [
                          {"name": "SEQ_SCAN ", "timing": 0.179, "cardinality": 416072,
                           "extra_info": "nesreca\n[INFOSEPARATOR]\ny\nx\ny_wgs84\n[INFOSEPARATOR]\nEC: 508993",
                           "timings": [], "children": []}]}]}]}
        expected_plan4 = {
            'plan_parameters': {
                'act_card': 416072,
                'act_children_card': 416072,
                'est_card': -1,
                'est_children_card': 508993,
                'op_name': 'PROJECTION',
                'udf_table': 1,
                'udf_params': [37, 23, 38],
                'output_columns': [
                    {'aggregation': 'None',
                     'child_ref': False,
                     'columns': None,
                     'udf_name': 'func_0',
                     'udf_output': 'True'}
                ]
            },
            'children': [
                {
                    'plan_parameters': {
                        'act_card': 416072,
                        'act_children_card': 1,
                        'est_card': 508993,
                        'est_children_card': 1,
                        'op_name': 'SEQ_SCAN',
                        'table': 1,
                        'table_name': 'nesreca',
                        'output_columns': [
                            {'aggregation': 'None', 'columns': [37], 'udf_name': None, 'child_ref': False,
                             'udf_output': 'False'},
                            {'aggregation': 'None', 'columns': [23], 'udf_name': None, 'child_ref': False,
                             'udf_output': 'False'},
                            {'aggregation': 'None', 'columns': [38], 'udf_name': None, 'child_ref': False,
                             'udf_output': 'False'},
                        ]
                    },
                    'children': []
                }
            ]
        }
        parsed_plan4 = parse_plan_recursively(
            json.loads(json.dumps(raw_plan4), object_hook=lambda d: SimpleNamespace(**d)),
            include_zero_card=True, table_id_mapping=table_id_mapping,
            column_id_mapping=column_id_mapping)
        self.maxDiff = None
        self.assertEqual(expected_plan4, parsed_plan4)

        raw_plan5 = {"name": "Query", "result": 0.044047, "timing": 0.044052, "cardinality": 0,
                     "extra-info": "EXPLAIN ANALYZE SELECT AVG(func_1(\"loan\".\"duration\")) as agg_0 FROM \"loan\" JOIN \"account\" ON \"loan\".\"account_id\" = \"account\".\"account_id\" JOIN \"orders\" ON \"account\".\"account_id\" = \"orders\".\"account_id\";",
                     "timings": [], "children": [
                {"name": "EXPLAIN_ANALYZE", "timing": 0.0, "cardinality": 0, "extra_info": "", "timings": [],
                 "children": [
                     {"name": "UNGROUPED_AGGREGATE", "timing": 2.8e-05, "cardinality": 1, "extra_info": "avg(#0)",
                      "timings": [], "children": [{"name": "PROJECTION", "timing": 0.024384, "cardinality": 2018,
                                                   "extra_info": "func_1(CAST(duration AS BIGINT))\n", "timings": [],
                                                   "children": [
                                                       {"name": "HASH_JOIN", "timing": 0.000394, "cardinality": 2018,
                                                        "extra_info": "INNER\naccount_id = account_id\n\n[INFOSEPARATOR]\nEC: 3922\n",
                                                        "timings": [], "children": [
                                                           {"name": "SEQ_SCAN ", "timing": 0.007901,
                                                            "cardinality": 21747,
                                                            "extra_info": "orders\n[INFOSEPARATOR]\naccount_id\n[INFOSEPARATOR]\nFilters: account_id>=2 AND account_id IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 25884",
                                                            "timings": [], "children": []},
                                                           {"name": "HASH_JOIN", "timing": 0.000346,
                                                            "cardinality": 1052,
                                                            "extra_info": "INNER\naccount_id = account_id\n\n[INFOSEPARATOR]\nEC: 2728\n",
                                                            "timings": [], "children": [
                                                               {"name": "SEQ_SCAN ", "timing": 0.009283,
                                                                "cardinality": 17998,
                                                                "extra_info": "account\n[INFOSEPARATOR]\naccount_id\n[INFOSEPARATOR]\nFilters: account_id>=2 AND account_id<=45512 AND account_id IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 18000",
                                                                "timings": [], "children": []},
                                                               {"name": "SEQ_SCAN ", "timing": 4.6e-05,
                                                                "cardinality": 2728,
                                                                "extra_info": "loan\n[INFOSEPARATOR]\naccount_id\nduration\n[INFOSEPARATOR]\nEC: 2728",
                                                                "timings": [], "children": []}]}]}]}]}]}]}
        expected_plan5 = {
            'plan_parameters': {
                'act_card': 1,
                'act_children_card': 2018,
                'est_card': -1,
                'est_children_card': 3922,
                'op_name': 'UNGROUPED_AGGREGATE',
                'udf_table': 8,
                'udf_params': [57],
                'output_columns': [
                    {'aggregation': 'AVG', 'columns': None, 'child_ref': False,
                     'udf_name': 'func_1', 'udf_output': 'True'}
                ]
            },
            'children': [
                {
                    'plan_parameters': {
                        'act_card': 2018,
                        'act_children_card': 22877844,
                        'est_card': 3922,
                        'est_children_card': 70611552,
                        'op_name': 'HASH_JOIN',
                        'join': 'INNER\naccount_id = account_id',
                    },
                    'children': [
                        {
                            'plan_parameters': {
                                'act_card': 21747,
                                'act_children_card': 1,
                                'est_card': 25884,
                                'est_children_card': 1,
                                'op_name': 'SEQ_SCAN',
                                'table': 6,
                                'table_name': 'orders',
                                'output_columns': [
                                    {'aggregation': 'None', 'columns': [54], 'udf_name': None, 'child_ref': False,
                                     'udf_output': 'False'}
                                ],
                                'filter_columns': PredicateNode(
                                    operator=LogicalOperator.AND,
                                    literal_feature=None,
                                    children=[
                                        PredicateNode(operator=Operator.GEQ, literal='2', column=54,
                                                      children=[], text=None, literal_feature=0),
                                        PredicateNode(operator=Operator.IS_NOT_NULL, column=54,
                                                      children=[], text=None, literal_feature=0, literal=''),
                                    ]
                                ),
                            },
                            'children': []
                        },
                        {
                            'plan_parameters': {
                                'act_card': 1052,
                                'act_children_card': 49098544,
                                'est_card': 2728,
                                'est_children_card': 49104000,
                                'op_name': 'HASH_JOIN',
                                'join': 'INNER\naccount_id = account_id',
                            }, 'children': [
                            {
                                'plan_parameters': {
                                    'act_card': 17998,
                                    'act_children_card': 1,
                                    'est_card': 18000,
                                    'est_children_card': 1,
                                    'op_name': 'SEQ_SCAN',
                                    'table': 7,
                                    'table_name': 'account',
                                    'output_columns': [
                                        {'aggregation': 'None', 'columns': [55], 'udf_name': None,
                                         'child_ref': False, 'udf_output': 'False'}
                                    ],
                                    'filter_columns': PredicateNode(
                                        operator=LogicalOperator.AND,
                                        children=[
                                            PredicateNode(operator=Operator.GEQ, literal='2',
                                                          column=55, children=[], text=None,
                                                          literal_feature=0),
                                            PredicateNode(operator=Operator.LEQ, literal='45512',
                                                          column=55, children=[], text=None,
                                                          literal_feature=0),
                                            PredicateNode(operator=Operator.IS_NOT_NULL,
                                                          column=55, children=[], text=None,
                                                          literal_feature=0, literal=''),
                                        ]
                                    ),
                                },
                                'children': [],
                            },
                            {
                                'plan_parameters': {
                                    'act_card': 2728,
                                    'act_children_card': 1,
                                    'est_card': 2728,
                                    'est_children_card': 1,
                                    'op_name': 'SEQ_SCAN',
                                    'table': 8,
                                    'table_name': 'loan',
                                    'output_columns': [
                                        {'aggregation': 'None', 'columns': [56], 'udf_name': None,
                                         'child_ref': False, 'udf_output': 'False'},
                                        {'aggregation': 'None', 'columns': [57], 'udf_name': None,
                                         'child_ref': False, 'udf_output': 'False'}
                                    ],

                                },
                                'children': []
                            }
                        ]
                        }
                    ]
                }
            ]
        }
        parsed_plan5 = parse_plan_recursively(
            json.loads(json.dumps(raw_plan5), object_hook=lambda d: SimpleNamespace(**d)),
            include_zero_card=True, table_id_mapping=table_id_mapping,
            column_id_mapping=column_id_mapping)
        self.maxDiff = None
        self.assertEqual(expected_plan5, parsed_plan5)

        raw_plan6 = {"name": "Query", "result": 0.114159, "timing": 0.114171, "cardinality": 0,
                     "extra-info": "EXPLAIN ANALYZE SELECT COUNT(*) as agg_0 FROM \"orders\" JOIN \"account\" ON \"orders\".\"account_id\" = \"account\".\"account_id\" JOIN \"district\" ON \"account\".\"district_id\" = \"district\".\"district_id\"  WHERE func_5(\"orders\".\"order_id\",\"orders\".\"account_to\",\"orders\".\"account_id\",\"orders\".\"bank_to\") >= 4648136 AND \"orders\".\"amount\" <= 2072.2637025930444;",
                     "timings": [], "children": [
                {"name": "EXPLAIN_ANALYZE", "timing": 0.0, "cardinality": 0, "extra_info": "", "timings": [],
                 "children": [
                     {"name": "UNGROUPED_AGGREGATE", "timing": 2.1e-05, "cardinality": 1, "extra_info": "count_star()",
                      "timings": [], "children": [{"name": "HASH_JOIN", "timing": 0.000411, "cardinality": 5905,
                                                   "extra_info": "INNER\ndistrict_id = district_id\n\n[INFOSEPARATOR]\nEC: 4195\n",
                                                   "timings": [], "children": [
                             {"name": "PROJECTION", "timing": 9e-06, "cardinality": 6047, "extra_info": "#1\n",
                              "timings": [], "children": [{"name": "FILTER", "timing": 0.000104, "cardinality": 6047,
                                                           "extra_info": "(district_id <= 311)\n[INFOSEPARATOR]\nEC: 5176",
                                                           "timings": [], "children": [
                                     {"name": "HASH_JOIN", "timing": 0.002048, "cardinality": 6058,
                                      "extra_info": "INNER\naccount_id = account_id\n\n[INFOSEPARATOR]\nEC: 5176\n",
                                      "timings": [], "children": [
                                         {"name": "SEQ_SCAN ", "timing": 0.000217, "cardinality": 17999,
                                          "extra_info": "account\n[INFOSEPARATOR]\naccount_id\ndistrict_id\n[INFOSEPARATOR]\nFilters: account_id<=45512 AND account_id IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 18000",
                                          "timings": [], "children": []},
                                         {"name": "FILTER", "timing": 0.109292, "cardinality": 7341,
                                          "extra_info": "((CAST(amount AS DECIMAL(18,13)) <= 2072.2637025930444) AND (func_5(CAST(order_id AS BIGINT), CAST(account_to AS BIGINT), CAST(account_id AS BIGINT), bank_to) >= 4648136))\n[INFOSEPARATOR]\nEC: 5176",
                                          "timings": [], "children": [
                                             {"name": "SEQ_SCAN ", "timing": 0.000657, "cardinality": 21747,
                                              "extra_info": "orders\n[INFOSEPARATOR]\naccount_id\norder_id\naccount_to\nbank_to\namount\n[INFOSEPARATOR]\nEC: 5176",
                                              "timings": [], "children": []}]}]}]}]},
                             {"name": "SEQ_SCAN ", "timing": 2.5e-05, "cardinality": 304,
                              "extra_info": "district\n[INFOSEPARATOR]\ndistrict_id\n[INFOSEPARATOR]\nEC: 308",
                              "timings": [], "children": []}]}]}]}]}

        parsed_plan6 = parse_plan_recursively(
            json.loads(json.dumps(raw_plan6), object_hook=lambda d: SimpleNamespace(**d)),
            include_zero_card=True, table_id_mapping=table_id_mapping,
            column_id_mapping=column_id_mapping)

        expected_plan6 = {
            'plan_parameters': {
                'act_card': 1,
                'act_children_card': 5905,
                'est_card': -1,
                'est_children_card': 4195,
                'op_name': 'UNGROUPED_AGGREGATE',
                'output_columns': [
                    {'aggregation': 'COUNT', 'columns': [], 'udf_name': None, 'child_ref': False,
                     'udf_output': 'False'}
                ]
            },
            'children': [
                {'plan_parameters': {
                    'act_card': 5905,
                    'act_children_card': 1838288,
                    'est_card': 4195,
                    'est_children_card': 1594208,
                    'op_name': 'HASH_JOIN',
                    'join': 'INNER\ndistrict_id = district_id',
                },
                    'children': [
                        {'plan_parameters': {
                            'op_name': 'FILTER',
                            'act_card': 6047,
                            'act_children_card': 6058,
                            'est_card': 5176,
                            'est_children_card': 5176,
                            'filter_columns': PredicateNode(operator=Operator.LEQ, literal='311',
                                                            column=58,
                                                            children=[], text=None, literal_feature=0),

                        },
                            'children': [
                                {'plan_parameters': {
                                    'op_name': 'HASH_JOIN',
                                    'act_card': 6058,
                                    'act_children_card': 132130659,
                                    'est_card': 5176,
                                    'est_children_card': 93168000,
                                    'join': 'INNER\naccount_id = account_id',

                                },
                                    'children': [
                                        {'plan_parameters': {
                                            'op_name': 'SEQ_SCAN',
                                            'act_card': 17999,
                                            'act_children_card': 1,
                                            'est_card': 18000,
                                            'est_children_card': 1,
                                            'table': 7,
                                            'table_name': 'account',
                                            'output_columns': [
                                                {'aggregation': 'None', 'columns': [55], 'udf_name': None,
                                                 'child_ref': False, 'udf_output': 'False'},
                                                {'aggregation': 'None', 'columns': [58], 'udf_name': None,
                                                 'child_ref': False, 'udf_output': 'False'}
                                            ],
                                            'filter_columns': PredicateNode(
                                                operator=LogicalOperator.AND,
                                                children=[
                                                    PredicateNode(operator=Operator.LEQ, literal='45512',
                                                                  column=55, children=[], text=None,
                                                                  literal_feature=0),
                                                    PredicateNode(operator=Operator.IS_NOT_NULL,
                                                                  column=55, children=[], text=None,
                                                                  literal_feature=0, literal=''),
                                                ]
                                            ),
                                        },
                                            'children': []
                                        },
                                        {'plan_parameters': {
                                            'op_name': 'FILTER',
                                            'act_card': 7341,
                                            'act_children_card': 21747,
                                            'est_card': 5176,
                                            'est_children_card': 5176,
                                            'filter_columns': PredicateNode(
                                                operator=LogicalOperator.AND,
                                                children=[
                                                    PredicateNode(operator=Operator.LEQ,
                                                                  literal='2072.2637025930444',
                                                                  column=62, children=[], text=None,
                                                                  literal_feature=0),
                                                    PredicateNode(operator=Operator.GEQ, literal='4648136',
                                                                  column=None, children=[], text=None,
                                                                  literal_feature=0, udf_name='func_5'),
                                                ]
                                            ),
                                            'udf_params': [59, 60, 54, 61],
                                            'udf_table': 6
                                        },
                                            'children': [
                                                {'plan_parameters': {
                                                    'op_name': 'SEQ_SCAN',
                                                    'act_card': 21747,
                                                    'act_children_card': 1,
                                                    'est_card': 5176,
                                                    'est_children_card': 1,
                                                    'table': 6,
                                                    'table_name': 'orders',
                                                    'output_columns': [
                                                        {'aggregation': 'None', 'columns': [54], 'udf_name': None,
                                                         'child_ref': False, 'udf_output': 'False'},
                                                        {'aggregation': 'None', 'columns': [59], 'udf_name': None,
                                                         'child_ref': False, 'udf_output': 'False'},
                                                        {'aggregation': 'None', 'columns': [60], 'udf_name': None,
                                                         'child_ref': False, 'udf_output': 'False'},
                                                        {'aggregation': 'None', 'columns': [61], 'udf_name': None,
                                                         'child_ref': False, 'udf_output': 'False'},
                                                        {'aggregation': 'None', 'columns': [62], 'udf_name': None,
                                                         'child_ref': False, 'udf_output': 'False'},

                                                    ]}, 'children': [], }]}
                                    ]
                                }
                            ]
                        },
                        {
                            'plan_parameters': {
                                'op_name': 'SEQ_SCAN',
                                'act_card': 304,
                                'act_children_card': 1,
                                'est_card': 308,
                                'est_children_card': 1,
                                'table': 9,
                                'table_name': 'district',
                                'output_columns': [
                                    {'aggregation': 'None', 'columns': [63], 'udf_name': None, 'child_ref': False,
                                     'udf_output': 'False'}
                                ],
                            },
                            'children': []
                        }
                    ]
                }
            ]
        }

        parsed_plan6 = parse_plan_recursively(
            json.loads(json.dumps(raw_plan6), object_hook=lambda d: SimpleNamespace(**d)),
            include_zero_card=True, table_id_mapping=table_id_mapping,
            column_id_mapping=column_id_mapping)
        self.maxDiff = None
        self.assertEqual(expected_plan6, parsed_plan6)

        raw_plan7 = {"name": "Query", "result": 0.102127, "timing": 0.102154, "cardinality": 0,
                     "extra-info": "EXPLAIN ANALYZE SELECT COUNT(*) FROM \"nation\" JOIN \"region\" ON \"nation\".\"n_regionkey\" = \"region\".\"r_regionkey\" JOIN \"supplier\" ON \"nation\".\"n_nationkey\" = \"supplier\".\"s_nationkey\" JOIN \"partsupp\" ON \"supplier\".\"s_suppkey\" = \"partsupp\".\"ps_suppkey\" JOIN \"part\" ON \"partsupp\".\"ps_partkey\" = \"part\".\"p_partkey\" JOIN \"lineitem\" ON \"partsupp\".\"ps_partkey\" = \"lineitem\".\"l_partkey\" AND \"partsupp\".\"ps_suppkey\" = \"lineitem\".\"l_suppkey\"  WHERE \"supplier\".\"s_acctbal\" <= 6648.755920172838 AND \"partsupp\".\"ps_supplycost\" >= 568.6510100445133 AND \"partsupp\".\"ps_availqty\" >= 2924 AND \"supplier\".\"s_suppkey\" <= 9430;",
                     "timings": [], "children": [
                {"name": "RESULT_COLLECTOR", "timing": 0.0, "cardinality": 0, "extra_info": "", "timings": [],
                 "children": [
                     {"name": "EXPLAIN_ANALYZE", "timing": 0.0, "cardinality": 0, "extra_info": "", "timings": [],
                      "children": [{"name": "UNGROUPED_AGGREGATE", "timing": 0.002707, "cardinality": 1,
                                    "extra_info": "count_star()", "timings": [], "children": [
                              {"name": "HASH_JOIN", "timing": 0.570661, "cardinality": 1203847,
                               "extra_info": "INNER\nl_partkey = ps_partkey\nl_suppkey = ps_suppkey\n\n[INFOSEPARATOR]\nEC: 18547\n",
                               "timings": [], "children": [
                                  {"name": "SEQ_SCAN ", "timing": 0.575987, "cardinality": 5658153,
                                   "extra_info": "lineitem\n[INFOSEPARATOR]\nl_partkey\nl_suppkey\n[INFOSEPARATOR]\nFilters: l_suppkey<=9430 AND l_suppkey IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 6001215",
                                   "timings": [], "children": []},
                                  {"name": "HASH_JOIN", "timing": 0.025676, "cardinality": 160527,
                                   "extra_info": "INNER\np_partkey = ps_partkey\n\n[INFOSEPARATOR]\nEC: 848\n",
                                   "timings": [], "children": [
                                      {"name": "SEQ_SCAN ", "timing": 0.000659, "cardinality": 200000,
                                       "extra_info": "part\n[INFOSEPARATOR]\np_partkey\n[INFOSEPARATOR]\nEC: 200000",
                                       "timings": [], "children": []},
                                      {"name": "HASH_JOIN", "timing": 0.003775, "cardinality": 160527,
                                       "extra_info": "INNER\nps_suppkey = s_suppkey\n\n[INFOSEPARATOR]\nEC: 1163\n",
                                       "timings": [], "children": [
                                          {"name": "FILTER", "timing": 0.022715, "cardinality": 231069,
                                           "extra_info": "(CAST(ps_supplycost AS DECIMAL(26,13)) >= 568.6510100445133)\n[INFOSEPARATOR]\nEC: 32000",
                                           "timings": [], "children": [
                                              {"name": "SEQ_SCAN ", "timing": 0.02504, "cardinality": 534749,
                                               "extra_info": "partsupp\n[INFOSEPARATOR]\nps_suppkey\nps_partkey\nps_supplycost\n[INFOSEPARATOR]\nFilters: ps_suppkey<=9430 AND ps_suppkey IS NOT NULL\nps_availqty>=2924 AND ps_availqty IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 32000",
                                               "timings": [], "children": []}]},
                                          {"name": "HASH_JOIN", "timing": 0.000251, "cardinality": 6550,
                                           "extra_info": "INNER\ns_nationkey = n_nationkey\n\n[INFOSEPARATOR]\nEC: 400\n",
                                           "timings": [], "children": [
                                              {"name": "FILTER", "timing": 0.00044, "cardinality": 6550,
                                               "extra_info": "(CAST(s_acctbal AS DECIMAL(25,12)) <= 6648.755920172838)\n[INFOSEPARATOR]\nEC: 400",
                                               "timings": [], "children": [
                                                  {"name": "SEQ_SCAN ", "timing": 0.000504, "cardinality": 9429,
                                                   "extra_info": "supplier\n[INFOSEPARATOR]\ns_nationkey\ns_suppkey\ns_acctbal\n[INFOSEPARATOR]\nFilters: s_suppkey<=9430 AND s_suppkey IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 400",
                                                   "timings": [], "children": []}]},
                                              {"name": "HASH_JOIN", "timing": 0.000174, "cardinality": 25,
                                               "extra_info": "INNER\nn_regionkey = r_regionkey\n\n[INFOSEPARATOR]\nEC: 25\n",
                                               "timings": [], "children": [
                                                  {"name": "SEQ_SCAN ", "timing": 3.5e-05, "cardinality": 25,
                                                   "extra_info": "nation\n[INFOSEPARATOR]\nn_regionkey\nn_nationkey\n[INFOSEPARATOR]\nEC: 25",
                                                   "timings": [], "children": []},
                                                  {"name": "SEQ_SCAN ", "timing": 0.000423, "cardinality": 5,
                                                   "extra_info": "region\n[INFOSEPARATOR]\nr_regionkey\n[INFOSEPARATOR]\nEC: 5",
                                                   "timings": [], "children": []}]}]}]}]}]}]}]}]}]}

        parsed_plan7 = parse_plan_recursively(
            json.loads(json.dumps(raw_plan7), object_hook=lambda d: SimpleNamespace(**d)),
            include_zero_card=True, table_id_mapping=table_id_mapping,
            column_id_mapping=column_id_mapping)

        raw_plan8 = {'name': 'Query', 'result': 0.052784, 'timing': 0.052809, 'cardinality': 0,
                     'extra-info': 'EXPLAIN ANALYZE SELECT * FROM (SELECT * FROM "movie_keyword" JOIN "keyword" ON "movie_keyword"."keyword_id" = "keyword"."id"  WHERE "movie_keyword"."id" <= 318267 AND "movie_keyword"."movie_id" >= 2092778) as nested_query WHERE func_3(nested_query."movie_id",nested_query."keyword_id") <= 947;',
                     'timings': [], 'children': [
                {'name': 'RESULT_COLLECTOR', 'timing': 0.0, 'cardinality': 0, 'extra_info': '', 'timings': [],
                 'children': [
                     {'name': 'EXPLAIN_ANALYZE', 'timing': 1e-06, 'cardinality': 0, 'extra_info': '', 'timings': [],
                      'children': [{'name': 'FILTER', 'timing': 0.001859, 'cardinality': 4,
                                    'extra_info': '(func_3(CAST(movie_id AS BIGINT), CAST(keyword_id AS BIGINT)) <= 947)\n[INFOSEPARATOR]\nEC: 4523930',
                                    'timings': [], 'children': [
                              {'name': 'PROJECTION', 'timing': 3e-06, 'cardinality': 5,
                               'extra_info': 'id\nmovie_id\nkeyword_id\nid\nkeyword\nphonetic_code\n', 'timings': [],
                               'children': [{'name': 'PROJECTION', 'timing': 5e-06, 'cardinality': 5,
                                             'extra_info': '#0\n#1\n#2\n#4\n#5\n', 'timings': [], 'children': [
                                       {'name': 'HASH_JOIN', 'timing': 0.018066, 'cardinality': 5,
                                        'extra_info': 'INNER\nkeyword_id = id\n\n[INFOSEPARATOR]\nBuild Min: 2\nBuild Max: 134170\n\n[INFOSEPARATOR]\nEC: 4523930\n',
                                        'timings': [], 'children': [
                                           {'name': 'SEQ_SCAN ', 'timing': 0.005446, 'cardinality': 5,
                                            'extra_info': 'movie_keyword\n[INFOSEPARATOR]\nkeyword_id\nid\nmovie_id\n[INFOSEPARATOR]\nFilters: keyword_id<=134170 AND keyword_id IS NOT NULL\nid<=318267 AND id IS NOT NULL\nmovie_id>=2092778 AND movie_id IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 4523930',
                                            'timings': [], 'children': []},
                                           {'name': 'SEQ_SCAN ', 'timing': 0.017701, 'cardinality': 134131,
                                            'extra_info': 'keyword\n[INFOSEPARATOR]\nid\nkeyword\nphonetic_code\n[INFOSEPARATOR]\nFilters: id>=2 AND id IS NOT NULL\n\n[INFOSEPARATOR]\nEC: 134132',
                                            'timings': [], 'children': []}]}]}]}]}]}]}]}

        parsed_plan8 = parse_plan_recursively(
            json.loads(json.dumps(raw_plan8), object_hook=lambda d: SimpleNamespace(**d)), include_zero_card=True,
            table_id_mapping=table_id_mapping, column_id_mapping=column_id_mapping)

    def test_parse_output_columns(self):
        out_cols_str = 'func_124(CAST(ps_suppkey AS BIGINT), CAST(ps_partkey AS BIGINT))'
        output_columns, udf_table_list, udf_params_lists = parse_output_columns(out_cols_str, column_id_mapping={
            ('test', 'ps_suppkey'): 42, ('test', 'ps_partkey'): 1}, table_id_mapping={'test': 11}, table_name='test')

        self.assertEqual([11], udf_table_list)
        self.assertEqual([[42, 1]], udf_params_lists)
        self.assertEqual([{'aggregation': 'None', 'columns': None, 'udf_name': 'func_124', 'udf_output': 'True',
                           'child_ref': False}],
                         output_columns)

        out_cols_str = '(CAST(amount AS DECIMAL(12,1)) + CAST(duration AS DECIMAL(12,1)))\n'
        output_columns, udf_table_list, udf_params_lists = parse_output_columns(out_cols_str, column_id_mapping={
            ('test', 'amount'): 42, ('test', 'duration'): 1}, table_id_mapping={'test': 11}, table_name='test')
        self.assertEqual([], udf_table_list)
        self.assertEqual([], udf_params_lists)
        self.assertEqual([{'aggregation': 'None', 'columns': [42, 1], 'udf_name': None, 'udf_output': 'False',
                           'child_ref': False}], output_columns)

        out_cols_str = 'avg_revenue\n'
        output_columns, udf_table_list, udf_params_lists = parse_output_columns(out_cols_str, column_id_mapping={
            ('test', 'avg_revenue'): 42}, table_id_mapping={'test': 11}, table_name='test')
        self.assertEqual([], udf_table_list)
        self.assertEqual([], udf_params_lists)
        self.assertEqual([{'aggregation': 'None', 'columns': [42], 'udf_name': None, 'udf_output': 'False',
                           'child_ref': False}], output_columns)

        out_cols_str = 'CAST((CAST(team_id1 AS FLOAT) + pred) AS DOUBLE)\n'
        output_columns, udf_table_list, udf_params_lists = parse_output_columns(out_cols_str, column_id_mapping={
            ('test', 'team_id1'): 1, ('test', 'pred'): 2}, table_id_mapping={'test': 11}, table_name='test')
        self.assertEqual([], udf_table_list)
        self.assertEqual([], udf_params_lists)
        self.assertEqual([{'aggregation': 'None', 'columns': [1, 2], 'udf_name': None, 'udf_output': 'False',
                           'child_ref': False}], output_columns)

    def test_remove_cast_nesting(self):
        str1 = 'CAST((CAST(team_id1 AS FLOAT) + pred) AS DOUBLE)'
        self.assertEqual('(team_id1 + pred)', remove_cast_nesting(str1)[0])

        str2 = '(CAST(amount AS DECIMAL(12,1)) + CAST(duration AS DECIMAL(12,1)))'
        self.assertEqual('(amount + duration)', remove_cast_nesting(str2)[0])

        str3 = '(CAST(min AS DECIMAL(22,12)) <= 19878.092123331367)'
        self.assertEqual('(min <= 19878.092123331367)', remove_cast_nesting(str3)[0])


if __name__ == '__main__':
    unittest.main()
