import unittest

from deepdb.inference import FilterCond

from udf_graph.annotate_graph_info import extract_where_conditions


class AnnotateGraphInfoTest(unittest.TestCase):
    def test_extract_where_conditions(self):
        sql1 = 'SELECT SUM("keyword"."id") as agg_0 FROM "movie_keyword" JOIN "keyword" ON "movie_keyword"."keyword_id" = "keyword"."id"  WHERE func_394("movie_keyword"."movie_id","movie_keyword"."keyword_id") >= 2420 AND "movie_keyword"."id" >= 258798 AND "keyword"."id" >= 78276;'
        self.assertEqual(
            [FilterCond(table_name='keyword', column_name='id', value='78276', operator='>=')],
            extract_where_conditions(sql1, 'keyword'))
        self.assertEqual([FilterCond(table_name='movie_keyword', column_name='id', value='258798', operator='>=')],
                         extract_where_conditions(sql1, 'movie_keyword'))
        self.assertEqual({FilterCond(table_name='keyword', column_name='id', value='78276', operator='>='),
                          FilterCond(table_name='movie_keyword', column_name='id', value='258798', operator='>=')},
                         set(extract_where_conditions(sql1, None)))

        sql2 = 'SELECT * FROM (SELECT * FROM "upravna_enota" JOIN "nesreca" ON "upravna_enota"."id_upravna_enota" = "nesreca"."upravna_enota" JOIN "oseba" ON "upravna_enota"."id_upravna_enota" = "oseba"."upravna_enota"  WHERE "nesreca"."tekst_odsek_ali_ulica" = \'POLULE\') as nested_query WHERE func_12(nested_query."povrsina",nested_query."id_upravna_enota",nested_query."ime_upravna_enota") != \'Žalec\';'

        self.assertEqual(
            [FilterCond(table_name='nesreca', column_name='tekst_odsek_ali_ulica', value='\'POLULE\'', operator='=')],
            extract_where_conditions(sql2, 'nesreca'))


if __name__ == '__main__':
    unittest.main()
