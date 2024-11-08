import json
import unittest
from types import SimpleNamespace

from cross_db_benchmark.benchmark_tools.generate_workload import Operator, LogicalOperator
from cross_db_benchmark.benchmark_tools.postgres.parse_filter import parse_filter, PredicateNode
from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_plans


class TestParsePlan(unittest.TestCase):
    def test_parse_plans(self):
        with open('test_plan.json', 'r') as f:
            run_stats = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        parsed_runs, stats = parse_plans(run_stats, min_runtime_ms=10, max_runtime_ms=10000,
                                         parse_baseline=True, cap_queries=100,
                                         parse_join_conds=True,
                                         include_zero_card=True, explain_only=False)

    def test_parse_filter(self):
        res1 = parse_filter('(nesreca.upravna_enota = upravna_enota.id_upravna_enota)')

        self.assertEqual(PredicateNode(operator=Operator.EQ, column=('nesreca', 'upravna_enota'),
                                       literal='upravna_enota.id_upravna_enota', children=[], text=None,
                                       filter_feature=0), res1)
        res2 = parse_filter("(vzrok_nesreca <> 'VO'::bpchar)")
        self.assertEqual(PredicateNode(operator=Operator.NEQ, column=('vzrok_nesreca',),
                                       literal='\'VO\'::bpchar', children=[], text=None,
                                       filter_feature=0), res2)
        res3a = parse_filter(
            "((func_5(tip_nesreca, stanje_povrsina_vozisce, stanje_vozisce, upravna_enota, id_nesreca, x_wgs84, x, y_wgs84))::numeric <= 388302.29069827584)")
        self.assertEqual(
            PredicateNode(operator=Operator.LEQ, column=None, literal='388302.29069827584', children=[], text=None,
                          udf_name='func_5', filter_feature=0)
            , res3a)

        res3b = parse_filter(
            "((vzrok_nesreca <> 'VO'::bpchar) AND ((func_5(tip_nesreca, stanje_povrsina_vozisce, stanje_vozisce, upravna_enota, id_nesreca, x_wgs84, x, y_wgs84))::numeric <= 388302.29069827584))")
        self.assertEqual(PredicateNode(operator=LogicalOperator.AND, column=None, literal=None, children=[
            PredicateNode(operator=Operator.NEQ, column=('vzrok_nesreca',),
                          literal='\'VO\'::bpchar', children=[], text=None, filter_feature=0),
            PredicateNode(operator=Operator.LEQ, column=None, literal='388302.29069827584', children=[], text=None,
                          udf_name='func_5', filter_feature=0)
        ], text=None), res3b)

        res4 = parse_filter(
            "((st_prebivalcev >= 18438) AND (povrsina = 346) AND ((func_27(povrsina, id_upravna_enota, ime_upravna_enota))::text = 'Šmarje pri Jelšah'::text))")
        self.assertEqual(PredicateNode(operator=LogicalOperator.AND, column=None, literal=None, children=[
            PredicateNode(operator=Operator.GEQ, column=('st_prebivalcev',), literal='18438', children=[], text=None,
                          filter_feature=0),
            PredicateNode(operator=Operator.EQ, column=('povrsina',), literal='346', children=[], text=None,
                          filter_feature=0),
            PredicateNode(operator=Operator.EQ, column=None, literal='\'Šmarje pri Jelšah\'::text', children=[],
                          text=None,
                          udf_name='func_27', filter_feature=0)
        ], text=None, filter_feature=0), res4)

    if __name__ == '__main__':
        unittest.main()
