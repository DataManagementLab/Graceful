import os
import time

import duckdb
import psycopg2

# rewrite dict
accidents_rewrite_dict = {
    'SELECT COUNT(*) FROM "nesreca" WHERE "nesreca"."stacionazna_ali_hisna_st" >= 41061.98479982046;': 'SELECT COUNT(*) FROM "nesreca" WHERE "nesreca"."stacionazna_ali_hisna_st" >= \'41061.98479982046\';',
    'SELECT COUNT(*) FROM "helper_view" WHERE x != 378087.0;': 'SELECT COUNT(*) FROM "helper_view" WHERE x != 378087;'
}

accidents_pessimistic_rewrite_dict = {
    'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota" JOIN "oseba" ON "upravna_enota"."id_upravna_enota" = "oseba"."upravna_enota"  WHERE "upravna_enota"."st_prebivalcev" <= 75059 AND "nesreca"."vzrok_nesreca" = \'HI\' AND "nesreca"."stanje_vozisce" != \'OS\' AND "nesreca"."opis_prizorisce" = \'Z\' AND "nesreca"."stacionazna_ali_hisna_st" <= 68730.08177798828;;': 'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota" JOIN "oseba" ON "upravna_enota"."id_upravna_enota" = "oseba"."upravna_enota"  WHERE "upravna_enota"."st_prebivalcev" <= 75059 AND "nesreca"."vzrok_nesreca" = \'HI\' AND "nesreca"."stanje_vozisce" != \'OS\' AND "nesreca"."opis_prizorisce" = \'Z\' AND "nesreca"."stacionazna_ali_hisna_st" <= \'68730.08177798828\';;',
    'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota"  WHERE "nesreca"."id_nesreca" >= 113383 AND "upravna_enota"."id_upravna_enota" <= 5527 AND "nesreca"."stacionazna_ali_hisna_st" <= 37817.86912426342 AND "upravna_enota"."ime_upravna_enota" != \'Velenje\';;': 'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota"  WHERE "nesreca"."id_nesreca" >= 113383 AND "upravna_enota"."id_upravna_enota" <= 5527 AND "nesreca"."stacionazna_ali_hisna_st" <= \'37817.86912426342\' AND "upravna_enota"."ime_upravna_enota" != \'Velenje\';;',
    'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota" JOIN "oseba" ON "upravna_enota"."id_upravna_enota" = "oseba"."upravna_enota"  WHERE "nesreca"."tekst_cesta_ali_naselje" = \'REČICA-ILIRSKA BISTRICA\' AND "nesreca"."stacionazna_ali_hisna_st" <= 21.026707299393657;;': 'CREATE VIEW helper_view AS SELECT "nesreca".* FROM "nesreca" JOIN "upravna_enota" ON "nesreca"."upravna_enota" = "upravna_enota"."id_upravna_enota" JOIN "oseba" ON "upravna_enota"."id_upravna_enota" = "oseba"."upravna_enota"  WHERE "nesreca"."tekst_cesta_ali_naselje" = \'REČICA-ILIRSKA BISTRICA\' AND "nesreca"."stacionazna_ali_hisna_st" <= \'21.026707299393657\';;'
}

import re


def modify_query(query):
    pattern = r'("nesreca"."stacionazna_ali_hisna_st"\s*(<=|>=|<|>|=|!=)\s*)(\d+(\.\d+)?)'
    replacement = r"\1'\3'"
    modified_query = re.sub(pattern, replacement, query)
    return modified_query

class DBMSWrapper():

    def __init__(self, dbms: str, dbms_kwargs: dict, db_name: str):
        super().__init__()
        self.dbms = dbms
        self.dbms_kwargs = dbms_kwargs
        self.db_name = db_name

        self.duck_imlp_casting = False

        if dbms == 'postgres':
            # Connect to your postgres DB
            self.conn = psycopg2.connect(database=db_name, host="localhost", user="postgres", port=dbms_kwargs['port'],
                                         password=dbms_kwargs['password'],
                                         options='-c statement_timeout=300000')
            # Open a cursor to perform database operations
            self.cur = self.conn.cursor()
        elif dbms == 'duckdb':
            import duckdb
            print(f'Duckdb version: {duckdb.__version__}')
            if 'version' not in dbms_kwargs:
                self.version = '0.10.1'
            else:
                self.version = dbms_kwargs["version"]

            assert duckdb.__version__ == self.version, f'Version mismatch: {duckdb.__version__} vs {self.version}'

            if self.version == '0.9.2':
                self.version_str = ''
            elif self.version == '0.10.1':
                self.version_str = '_10_1'
            else:
                raise ValueError(f"Unknown version {self.version}")

            if 'act_impl_casting' in dbms_kwargs and dbms_kwargs['act_impl_casting'] == 'True':
                self.duck_imlp_casting = True

            db_path = os.path.join(dbms_kwargs["dir"], f'{db_name}{self.version_str}.db')
            try:
                assert os.path.exists(db_path), f"Database {db_path} does not exist"
                self.conn = duckdb.connect(database=db_path,
                                           read_only=False)
            except duckdb.IOException as e:
                # wait for a second and then try again
                time.sleep(1)
                assert os.path.exists(db_path), f"Database {db_path} does not exist"
                self.conn = duckdb.connect(database=db_path,
                                           read_only=False)

            except Exception as e:
                print(f'Could not connect to {dbms_kwargs["dir"]}/{db_name}{self.version_str}.db')
                print(e)
                self.conn = None
            if self.duck_imlp_casting:
                print('activating old_implicit_casting')
                self.exec('SET old_implicit_casting = true;')
        else:
            raise NotImplementedError(f'Unknown dbms: {dbms}')

    def exec(self, sql: str):
        if self.db_name == 'accidents':
            sql = modify_query(sql)

        if self.dbms == 'postgres':
            self.cur.execute(sql)
            result = self.cur.fetchall()
            self.conn.commit()
        elif self.dbms == 'duckdb':
            try:
                result = self.conn.sql(sql)
            except duckdb.CatalogException as e:
                print(f'Could not execute {sql}')
                raise e
            except duckdb.ParserException as e:
                print(f'Could not parse {sql}')
                raise e
            except duckdb.BinderException as e:
                print(f'Could not bind {sql}')
                raise e
            if result is not None:
                try:
                    result = result.fetchall()
                except Exception as e:
                    if self.db_name == 'accidents' and sql in accidents_rewrite_dict:
                        try:
                            t_sql = accidents_rewrite_dict[sql]
                            result = self.conn.sql(t_sql).fetchall()
                        except Exception as e:
                            print(f'Could not fetch {sql}\nTranslated to {t_sql}')
                            raise e
                    else:
                        print(f'Could not fetch {sql}')
                        raise e
        else:
            raise NotImplementedError(f'Unknown dbms: {self.dbms}')
        return result

    def get_est_card(self, sql: str) -> int:
        explain_sql = f'EXPLAIN {sql}'
        if self.dbms == 'postgres':
            row = self.exec(explain_sql)[0][0]
            card = int(row.split('rows=')[1].split(' ')[0])
        elif self.dbms == 'duckdb':
            card = None
            row = self.exec(explain_sql)[0]

            empty_result_detected = False

            for block in row:
                regex = r'EC: ([0-9]+)'
                match = re.search(regex, block)
                if match is not None:
                    card = int(match.groups()[0])
                    break

                if 'EMPTY_RESULT' in block:
                    empty_result_detected = True
                    break

            if empty_result_detected:
                card = 0

            if card is None:
                # check whether the query is only a scan without projections, joins or filters
                if 'SELECT * FROM ' in sql:
                    card = self.get_est_card(sql.replace('SELECT * FROM ', 'SELECT COUNT(*) FROM '))
                    if card is not None:
                        return card

                print(f'Could not find estimated cardinality for {sql}\noutput: {row}')
                card = -1

        else:
            raise NotImplementedError(f'Unknown dbms: {self.dbms}')
        return card
