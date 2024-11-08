import unittest

from cross_db_benchmark.benchmark_tools.generate_workload import LogicalOperator, Operator
from cross_db_benchmark.benchmark_tools.postgres.parse_filter import parse_filter, PredicateNode


class ParseFilterTest(unittest.TestCase):
    def test_parse_filter(self):
        filter = parse_filter('(t.wfta>=24)', duckdb=True, sql='')
        expected_filter = PredicateNode(operator=Operator.GEQ, column=('t', 'wfta'), literal='24', literal_feature=None)

        self.assertEqual(expected_filter, filter)  # add assertion here

        filter = parse_filter('(wfta>=24 AND wfta IS NOT NULL AND wteam<=25856 AND wteam IS NOT NULL)', duckdb=True,
                              sql='')
        expected_filter = PredicateNode(
            operator=LogicalOperator.AND,
            children=[
                PredicateNode(operator=Operator.GEQ, column=('wfta',), literal='24', literal_feature=0),
                PredicateNode(operator=Operator.IS_NOT_NULL, column=('wfta',), literal='', literal_feature=0),
                PredicateNode(operator=Operator.LEQ, column=('wteam',), literal='25856', literal_feature=0),
                PredicateNode(operator=Operator.IS_NOT_NULL, column=('wteam',), literal='', literal_feature=0),
            ]
        )
        self.assertEqual(expected_filter, filter)  # add assertion here

        filter3 = parse_filter(
            '(((CAST(amount AS DECIMAL(18,13)) <= 2072.2637025930444) AND (func_5(CAST(order_id AS BIGINT), CAST(account_to AS BIGINT), CAST(account_id AS BIGINT), bank_to) >= 4648136)))',
            sql='',
            duckdb=True)
        expected_filter3 = PredicateNode(
            operator=LogicalOperator.AND,
            children=[
                PredicateNode(
                    operator=Operator.LEQ,
                    column=('amount',),
                    literal='2072.2637025930444',
                    literal_feature=0
                ),
                PredicateNode(
                    operator=Operator.GEQ,
                    column=None,
                    literal='4648136',
                    literal_feature=0,
                    udf_name='func_5',

                ),
            ]
        )
        self.assertEqual(expected_filter3, filter3)

        filter4 = parse_filter('((func_14(CAST(lscore AS BIGINT), CAST(wfga3 AS BIGINT), CAST(lpf AS BIGINT)) != 41))',
                               duckdb=True, sql='')
        expected_filter4 = PredicateNode(
            operator=Operator.NEQ,
            column=None,
            literal='41',
            literal_feature=0,
            udf_name='func_14',
        )
        self.assertEqual(expected_filter4, filter4)

        filter5 = parse_filter('(account_id BETWEEN 2 AND 45512)', duckdb=True, sql='')
        expected_filter5 = PredicateNode(
            operator=LogicalOperator.AND,
            children=[
                PredicateNode(
                    operator=Operator.GEQ,
                    column=('account_id',),
                    literal='2',
                    literal_feature=0
                ),
                PredicateNode(
                    operator=Operator.LEQ,
                    column=('account_id',),
                    literal='45512',
                    literal_feature=0
                ),
            ]
        )
        self.assertEqual(expected_filter5, filter5)

        filter6 = parse_filter(
            "((((CAST(kc_proklikano AS DECIMAL(21,13)) >= 540.3809297076131) AND (month_year_datum_transakce != '2013-05-01') AND constant_or_null(true, CAST(client_id AS DECIMAL(19,9)), 10768319.628346074) AND (func_97(CAST(client_id AS BIGINT), sluzba) != 'e'))))",
            sql='',
            duckdb=True)
        expected_filter6 = PredicateNode(
            operator=LogicalOperator.AND,
            children=[
                PredicateNode(
                    operator=Operator.GEQ,
                    column=('kc_proklikano',),
                    literal='540.3809297076131',
                    literal_feature=0
                ),
                PredicateNode(
                    operator=Operator.NEQ,
                    column=('month_year_datum_transakce',),
                    literal="'2013-05-01'",
                    literal_feature=0
                ),
                PredicateNode(
                    operator=Operator.NEQ,
                    column=None,
                    literal="'e'",
                    literal_feature=0,
                    udf_name='func_97',
                ),
            ]
        )
        self.assertEqual(expected_filter6, filter6)

        filter7 = parse_filter(
            "(((month_year_datum_transakce != '2013-04-01') AND constant_or_null(true, CAST(client_id AS DECIMAL(18,8)), 16680751.35821165)))",
            sql='',
            duckdb=True)
        expected_filter7 = PredicateNode(
            operator=Operator.NEQ,
            column=('month_year_datum_transakce',),
            literal="'2013-04-01'",
            literal_feature=0
        )
        self.assertEqual(expected_filter7, filter7)

        filter8 = parse_filter(
            '(CAST(min AS DECIMAL(22,12)) <= 19878.092123331367)',
            sql='',
            duckdb=True)
        expected_filter8 = PredicateNode(
            operator=Operator.LEQ,
            column=('min',),
            literal='19878.092123331367',
            literal_feature=0
        )
        self.assertEqual(expected_filter8, filter8)

        filter9 = parse_filter(
            '(draftFrom=(UCLA redshirt) Floridians AND draftFrom IS NOT NULL)',
            sql='',
            duckdb=True)
        expected_filter9 = PredicateNode(
            operator=LogicalOperator.AND,
            children=[
                PredicateNode(
                    operator=Operator.EQ,
                    column=('draftFrom',),
                    literal="'(UCLA redshirt) Floridians'",
                    literal_feature=0
                ),
                PredicateNode(
                    operator=Operator.IS_NOT_NULL,
                    column=('draftFrom',),
                    literal='',
                    literal_feature=0
                ),
            ]
        )
        self.maxDiff = None
        self.assertEqual(expected_filter9, filter9)

        filter10 = parse_filter(
            '(CAST(func_187(CAST(A13 AS DOUBLE), CAST(A16 AS BIGINT), CAST(A5 AS BIGINT), CAST(A15 AS BIGINT)) AS DECIMAL(32,13)) <= 1465.0228947870305)',
            sql='(func_187(A13, A16, A5, A15) <= 1465.0228947870305)',
            duckdb=True)
        expected_filter10 = PredicateNode(
            operator=Operator.LEQ,
            column=None,
            literal='1465.0228947870305',
            literal_feature=0,
            udf_name='func_187',
        )
        self.assertEqual(expected_filter10, filter10)

        filter11 = parse_filter(
            '(charge != \'a0=-inf<x<=-0_1355\')',
            sql='(charge=\'a0=-inf<x<=-0_1355\')',
            duckdb=True)
        expected_filter11 = PredicateNode(
            operator=Operator.NEQ,
            column=('charge',),
            literal="'a0=-inf<x<=-0_1355'",
            literal_feature=0
        )
        self.assertEqual(expected_filter11, filter11)

        filter12 = parse_filter(
            '(charge=a0=0_0615<x<=0_1375)',
            duckdb=True, sql='(charge=\'a0=0_0615<x<=0_1375\')')
        expected_filter12 = PredicateNode(
            operator=Operator.EQ,
            column=('charge',),
            literal="'a0=0_0615<x<=0_1375'",
            literal_feature=0
        )
        self.assertEqual(expected_filter12, filter12)

        # filter13 = parse_filter(
        #     '(nazev_organu_en=South-East European States (Republic of Albania, Bosnia and Herzegovina, Montenegro, Republic of Kosovo, Republic of Macedonia) AND nazev_organu_en IS NOT NULL)',
        #     duckdb=True)
        # expected_filter13 = PredicateNode(
        #     operator=LogicalOperator.AND,
        #     children=[
        #         PredicateNode(
        #             operator=Operator.EQ,
        #             column=('nazev_organu_en',),
        #             literal="'South-East European States (Republic of Albania, Bosnia and Herzegovina, Montenegro, Republic of Kosovo, Republic of Macedonia)'",
        #             literal_feature=0
        #         ),
        #         PredicateNode(
        #             operator=Operator.IS_NOT_NULL,
        #             column=('nazev_organu_en',),
        #             literal='',
        #             literal_feature=0
        #         ),
        #     ]
        # )
        # self.assertEqual(expected_filter13, filter13)

        # filter14 = parse_filter(
        #     '( uplny_naz=Rozhodnutí vlády o souhlasu s přelety a průjezdy ozbrojených sil členských států Organizace Severoatlantické smlouvy (NATO) a států zúčastněných v programu Partnerství pro mír (PfP) a s přelety ozbrojených sil Bosny a Hercegoviny, Srbska a Černé Hory, Státu Izrael, Jordánského hášimovského království, Egyptské arabské republiky, Království Saúdské Arábie, Státu Kuvajt, Ománu, Spojených arabských emirátů, Bahrajnského království, Syrské arabské republiky, Pákistánské islámské republiky, Alžírské demokratické a lidové republiky, Tuniské republiky, Čadské republiky a Organizace Severoatlantické smlouvy (NATO) přes území České republiky v době od 1. ledna do 31. prosince 2004, na které se vztahuje rozhodovací pravomoc vlády ve smyslu čl. 43 odst. 5 písm. a) ústavního zákona č. 1/1993 Sb., Ústava České republiky, ve znění ústavního zákona č. 300/2000 Sb. AND uplny_naz IS NOT NULL )',
        #     duckdb=True)

        filter15 = parse_filter(
            '(Description=Indiana)', duckdb=True, sql='(Description=\'Indiana\')'
        )
        expected_filter15 = PredicateNode(
            operator=Operator.EQ,
            column=('Description',),
            literal="'Indiana'",
            literal_feature=0
        )
        self.assertEqual(expected_filter15, filter15)

        filter16 = parse_filter(
            '(Description=Indiana AND Description IS NOT NULL)', duckdb=True,
            sql='(Description=\'Indiana\' AND Description IS NOT NULL)'

        )
        expected_filter16 = PredicateNode(
            operator=LogicalOperator.AND,
            children=[
                PredicateNode(
                    operator=Operator.EQ,
                    column=('Description',),
                    literal="'Indiana'",
                    literal_feature=0
                ),
                PredicateNode(
                    operator=Operator.IS_NOT_NULL,
                    column=('Description',),
                    literal='',
                    literal_feature=0
                ),
            ]
        )
        self.assertEqual(expected_filter16, filter16)

        filter17 = parse_filter(
            '(n_comment=hely enticingly express accounts. even, final  AND n_comment IS NOT NULL)', duckdb=True,
            sql='("nation"."n_comment" = \'hely enticingly express accounts. even, final \' AND "supplier"."s_address" != \'QCBRQcSyomEn7G78pdsb3o,0\')')
        expected_filter17 = PredicateNode(
            operator=LogicalOperator.AND,
            children=[
                PredicateNode(
                    operator=Operator.EQ,
                    column=('n_comment',),
                    literal="'hely enticingly express accounts. even, final '",
                    literal_feature=0
                ),
                PredicateNode(
                    operator=Operator.IS_NOT_NULL,
                    column=('n_comment',),
                    literal='',
                    literal_feature=0,
                ),
            ]
        )
        self.assertEqual(expected_filter17, filter17)


if __name__ == '__main__':
    unittest.main()
