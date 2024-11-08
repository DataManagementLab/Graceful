import unittest

from udf_graph.dbms_wrapper import modify_query


class DBMSWrapperTest(unittest.TestCase):
    def test_modify_query(self):
        self.maxDiff = None
        q = 'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota" JOIN "oseba" ON "upravna_enota"."id_upravna_enota" = "oseba"."upravna_enota"  WHERE "upravna_enota"."st_prebivalcev" <= 75059 AND "nesreca"."vzrok_nesreca" = \'HI\' AND "nesreca"."stanje_vozisce" != \'OS\' AND "nesreca"."opis_prizorisce" = \'Z\' AND "nesreca"."stacionazna_ali_hisna_st" <= 68730.08177798828;;'
        modified_query = modify_query(q)
        self.assertEqual(
            'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota" JOIN "oseba" ON "upravna_enota"."id_upravna_enota" = "oseba"."upravna_enota"  WHERE "upravna_enota"."st_prebivalcev" <= 75059 AND "nesreca"."vzrok_nesreca" = \'HI\' AND "nesreca"."stanje_vozisce" != \'OS\' AND "nesreca"."opis_prizorisce" = \'Z\' AND "nesreca"."stacionazna_ali_hisna_st" <= \'68730.08177798828\';;',
            modified_query)

        q2 = 'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota"  WHERE "nesreca"."id_nesreca" >= 113383 AND "upravna_enota"."id_upravna_enota" <= 5527 AND "nesreca"."stacionazna_ali_hisna_st" <= 37817.86912426342 AND "upravna_enota"."ime_upravna_enota" != \'Velenje\';;'
        modified_query2 = modify_query(q2)
        self.assertEqual(
            'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota"  WHERE "nesreca"."id_nesreca" >= 113383 AND "upravna_enota"."id_upravna_enota" <= 5527 AND "nesreca"."stacionazna_ali_hisna_st" <= \'37817.86912426342\' AND "upravna_enota"."ime_upravna_enota" != \'Velenje\';;',
            modified_query2
        )


if __name__ == '__main__':
    unittest.main()
