import unittest

from cross_db_benchmark.benchmark_tools.utils import pullup_udf_in_sql, rewrite_udf_to_nestsql, \
    rewrite_select_to_nestsql, is_join_cond, parse_join_cond, extract_table_from_udf_filter, extract_table_from_filter


class UtilsTest(unittest.TestCase):
    def test_pullup_udf_in_sql(self):
        query = 'SELECT COUNT(*) FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995 AND func_13(title.id) > 10;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT * FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995) as nested_query WHERE func_13(nested_query.id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query)[0])

        query = 'SELECT COUNT(*) FROM title,movie_keyword WHERE func_13(title.id) > 10 AND title.id=movie_keyword.movie_id AND title.production_year > 1995;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT * FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995) as nested_query WHERE func_13(nested_query.id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query)[0])

        query = 'SELECT COUNT(*) FROM title,movie_keyword WHERE func_13(title.id) > 10;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT * FROM title,movie_keyword) as nested_query WHERE func_13(nested_query.id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query)[0])

        query = 'SELECT sum(title.id) FROM title,movie_keyword WHERE func_13(title.id) > 10;'
        expected_query = 'SELECT SUM(nested_query.id) FROM (SELECT * FROM title,movie_keyword) as nested_query WHERE func_13(nested_query.id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query)[0])

        # intermed position - easy cases - equal to complete pullup (only 1 join - no intermed pos exists)
        query = 'SELECT COUNT(*) FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995 AND func_13(title.id) > 10;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT title.id AS title_id FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995) as nested_query WHERE func_13(nested_query.title_id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query, intermed_pos=True)[0])

        query = 'SELECT COUNT(*) FROM title,movie_keyword WHERE func_13(title.id) > 10 AND title.id=movie_keyword.movie_id AND title.production_year > 1995;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT title.id AS title_id FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995) as nested_query WHERE func_13(nested_query.title_id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query, intermed_pos=True)[0])

        query = 'SELECT sum(title.prod_year) FROM title,movie_keyword WHERE func_13(title.id) > 10 AND title.id=movie_keyword.movie_id AND title.production_year > 1995;'
        expected_query = 'SELECT SUM(nested_query.title_prod_year) FROM (SELECT title.prod_year AS title_prod_year,title.id AS title_id FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995) as nested_query WHERE func_13(nested_query.title_id) > 10;'

        self.assertEqual(expected_query, pullup_udf_in_sql(query, intermed_pos=True)[0])

        query = 'SELECT COUNT(*) FROM title,movie_keyword WHERE func_13(title.id) > 10;'

        with self.assertRaises(AssertionError):
            pullup_udf_in_sql(query, intermed_pos=True)

        query = 'SELECT sum(title.id) FROM title,movie_keyword WHERE func_13(title.id) > 10;'
        with self.assertRaises(AssertionError):
            pullup_udf_in_sql(query, intermed_pos=True)[0]

        # intermed position - real cases with 2 and 3 and 4 joins
        query = 'SELECT COUNT(*) FROM title,movie_keyword,aka_name WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995 AND func_13(title.tid) > 10 AND aka_name.id = title.id;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT title.tid AS title_tid,title.id AS title_id FROM title,movie_keyword WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995) as nested_query,aka_name WHERE func_13(nested_query.title_tid) > 10 AND aka_name.id=nested_query.title_id;'
        self.assertEqual(expected_query, pullup_udf_in_sql(query, intermed_pos=True, random_seed=1)[0])

        # 3 joins
        query = 'SELECT COUNT(*) FROM title,movie_keyword,aka_name,aka_title WHERE title.id=movie_keyword.movie_id AND title.production_year > 1995 AND func_13(title.id) > 10 AND aka_name.id = title.id AND aka_title.id = title.id;'
        expected_query = 'SELECT COUNT(*) FROM (SELECT title.id AS title_id FROM title,aka_name,movie_keyword WHERE title.id=movie_keyword.movie_id AND aka_name.id=title.id AND title.production_year > 1995) as nested_query,aka_title WHERE func_13(nested_query.title_id) > 10 AND aka_title.id=nested_query.title_id;'
        self.assertEqual(expected_query, pullup_udf_in_sql(query, intermed_pos=True, random_seed=0)[0])

        self.maxDiff=None
        query = 'SELECT * FROM "L_AIRLINE_ID" JOIN "On_Time_On_Time_Performance_2016_1" ON "L_AIRLINE_ID"."Code" = "On_Time_On_Time_Performance_2016_1"."AirlineID" JOIN "L_MONTHS" ON "On_Time_On_Time_Performance_2016_1"."Month" = "L_MONTHS"."Code" JOIN "L_CITY_MARKET_ID" ON "On_Time_On_Time_Performance_2016_1"."DestCityMarketID" = "L_CITY_MARKET_ID"."Code" JOIN "L_AIRPORT_SEQ_ID" ON "On_Time_On_Time_Performance_2016_1"."OriginAirportSeqID" = "L_AIRPORT_SEQ_ID"."Code"  WHERE func_0("L_AIRLINE_ID"."Description") = \'Latcr: LCQ\' AND "L_CITY_MARKET_ID"."Description" != \'Altai, Mongolia\' AND "L_CITY_MARKET_ID"."Code" >= 31855 AND "L_MONTHS"."Description" != \'October\''
        expected_query = ('SELECT * FROM ('
                          'SELECT L_AIRLINE_ID.Description AS L_AIRLINE_ID_Description,On_Time_On_Time_Performance_2016_1.DestCityMarketID AS On_Time_On_Time_Performance_2016_1_DestCityMarketID,On_Time_On_Time_Performance_2016_1.OriginAirportSeqID AS On_Time_On_Time_Performance_2016_1_OriginAirportSeqID FROM L_AIRLINE_ID,On_Time_On_Time_Performance_2016_1,L_MONTHS WHERE L_AIRLINE_ID.Code=On_Time_On_Time_Performance_2016_1.AirlineID AND On_Time_On_Time_Performance_2016_1.Month=L_MONTHS.Code AND "L_MONTHS"."Description" != \'October\''
                          ') as nested_query,L_AIRPORT_SEQ_ID,L_CITY_MARKET_ID WHERE func_0(nested_query.L_AIRLINE_ID_Description) = \'Latcr: LCQ\' AND nested_query.On_Time_On_Time_Performance_2016_1_DestCityMarketID=L_CITY_MARKET_ID.Code AND nested_query.On_Time_On_Time_Performance_2016_1_OriginAirportSeqID=L_AIRPORT_SEQ_ID.Code AND "L_CITY_MARKET_ID"."Description" != \'Altai, Mongolia\' AND "L_CITY_MARKET_ID"."Code" >= 31855;')
        self.assertEqual(expected_query, pullup_udf_in_sql(query, intermed_pos=True, random_seed=0)[0])

        query = 'SELECT COUNT(func_334("L_ONTIME_DELAY_GROUPS"."Code","L_ONTIME_DELAY_GROUPS"."Code")) as agg_0 FROM "L_ONTIME_DELAY_GROUPS" JOIN "On_Time_On_Time_Performance_2016_1" ON "L_ONTIME_DELAY_GROUPS"."Code" = "On_Time_On_Time_Performance_2016_1"."ArrivalDelayGroups" JOIN "L_STATE_FIPS" ON "On_Time_On_Time_Performance_2016_1"."DestStateFips" = "L_STATE_FIPS"."Code" JOIN "L_STATE_ABR_AVIATION" ON "On_Time_On_Time_Performance_2016_1"."DestState" = "L_STATE_ABR_AVIATION"."Code"  WHERE "On_Time_On_Time_Performance_2016_1"."Div1TailNum" = \'N425LV\' AND "L_STATE_ABR_AVIATION"."Description" != \'Mississippi\' AND "On_Time_On_Time_Performance_2016_1"."Year" != 2019 AND "L_STATE_FIPS"."Description" != \'Howland Island\';'
        # expected_query = 'SELECT COUNT(func_334(nested_query.L_ONTIME_DELAY_GROUPS_Code)) as agg_0 FROM (SELECT L_ONTIME_DELAY_GROUPS.Code AS L_ONTIME_DELAY_GROUPS_Code,On_Time_On_Time_Performance_2016_1.Div1TailNum AS On_Time_On_Time_Performance_2016_1_Div1TailNum,On_Time_On_Time_Performance_2016_1.Year AS On_Time_On_Time_Performance_2016_1_Year FROM L_ONTIME_DELAY_GROUPS,On_Time_On_Time_Performance_2016_1 WHERE L_ONTIME_DELAY_GROUPS.Code=On_Time_On_Time_Performance_2016_1.ArrivalDelayGroups AND "On_Time_On_Time_Performance_2016_1"."Year" != 2019) as nested_query,L_STATE_FIPS,L_STATE_ABR_AVIATION WHERE nested_query.On_Time_On_Time_Performance_2016_1_Div1TailNum=\'N425LV\' AND nested_query.On_Time_On_Time_Performance_2016_1_Div1TailNum=L_STATE_FIPS.Code AND nested_query.On_Time_On_Time_Performance_2016_1_Div1TailNum=L_STATE_ABR_AVIATION.Code AND "L_STATE_ABR_AVIATION"."Description" != \'Mississippi\' AND "L_STATE_FIPS"."Description" != \'Howland Island\';'
        self.assertFalse(pullup_udf_in_sql(query, intermed_pos=True, random_seed=0)[1])
        # self.assertEqual(expected_query, pullup_udf_in_sql(query, intermed_pos=True, random_seed=0)[0])

    def test_rewrite_udf_to_nestsql(self):
        self.assertEqual('func_80(nested_query."id",nested_query."person_id")',
                         rewrite_udf_to_nestsql('func_80("person_info"."id","person_info"."person_id")'))

    def test_rewrite_select_to_nestsql(self):
        self.assertEqual('nested_query."id",nested_query."person_id"',
                         rewrite_select_to_nestsql('"person_info"."id","person_info"."person_id"'))
        self.assertEqual('AVG(nested_query."id")', rewrite_select_to_nestsql('AVG("person_info"."id")'))
        self.assertEqual('SUM(nested_query."id" + nested_query."person_id")',
                         rewrite_select_to_nestsql('SUM("aka_name"."id" + "aka_name"."person_id")'))


    def test_is_join_cond(self):
        self.assertTrue(is_join_cond('title.id=movie_keyword.movie_id'))
        self.assertTrue(is_join_cond('title.id = movie_keyword.movie_id'))
        self.assertFalse(is_join_cond('title.id="movie_keyword.movie_id"'))

    def test_parse_join_cond(self):
        self.assertEqual(('title', 'id', 'movie_keyword', 'movie_id'), parse_join_cond('title.id=movie_keyword.movie_id'))
        self.assertEqual(('title', 'id', 'movie_keyword', 'movie_id'), parse_join_cond('title.id = movie_keyword.movie_id'))

    def test_extract_table_from_udf_filter(self):
        self.assertEqual('person_info', extract_table_from_udf_filter('func_80("person_info"."id","person_info"."person_id")'))
        self.assertEqual('aka_name', extract_table_from_udf_filter('func_80("aka_name"."id","aka_name"."person_id")'))

        # no "
        self.assertEqual('person_info', extract_table_from_udf_filter('func_80(person_info.id,person_info.person_id)'))

    def test_extract_table_from_filter(self):
        self.assertEqual('title', extract_table_from_filter('title.id=3'))
        self.assertEqual('title', extract_table_from_filter('title.id < "test.123>"'))
        self.assertEqual('title', extract_table_from_filter('"title".id = "test.123>"'))


if __name__ == '__main__':
    unittest.main()
