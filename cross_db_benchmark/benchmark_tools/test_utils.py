import unittest

from cross_db_benchmark.benchmark_tools.utils import pullup_udf_in_sql, rewrite_udf_to_nestsql, \
    rewrite_select_to_nestsql


class UtilsTest(unittest.TestCase):
    def test_pullup_udf_in_sql(self):
        query = 'SELECT COUNT(*) FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995 AND func_13(title.id) > 10;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT * FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995) as v WHERE func_13(v.id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query)[0])

        query = 'SELECT COUNT(*) FROM title,movie_keyword WHERE func_13(title.id) > 10 AND title.id=movie_keyword.movie_id AND title.production_year > 1995;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT * FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995) as v WHERE func_13(v.id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query)[0])

        query = 'SELECT COUNT(*) FROM title,movie_keyword WHERE func_13(title.id) > 10;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT * FROM title,movie_keyword) as v WHERE func_13(v.id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query)[0])

        query = 'SELECT sum(title.id) FROM title,movie_keyword WHERE func_13(title.id) > 10;'
        expected_query = 'SELECT SUM(v.id) FROM (SELECT * FROM title,movie_keyword) as v WHERE func_13(v.id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query)[0])

    def test_rewrite_udf_to_nestsql(self):
        self.assertEqual('func_80(v."id",v."person_id")',
                         rewrite_udf_to_nestsql('func_80("person_info"."id","person_info"."person_id")'))

    def test_rewrite_select_to_nestsql(self):
        self.assertEqual('v."id",v."person_id"',
                         rewrite_select_to_nestsql('"person_info"."id","person_info"."person_id"'))
        self.assertEqual('AVG(v."id")', rewrite_select_to_nestsql('AVG("person_info"."id")'))
        self.assertEqual('SUM(v."id" + v."person_id")',
                         rewrite_select_to_nestsql('SUM("aka_name"."id" + "aka_name"."person_id")'))


if __name__ == '__main__':
    unittest.main()
