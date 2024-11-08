import json
import os
from typing import List, Dict, Any

import pandas as pd

from cross_db_benchmark.datasets.datasets import dataset_list_dict
from udf_graph.dbms_wrapper import DBMSWrapper

skip_col_list = [(
    ('accidents', 'oseba', 'alkotest',),
    ('accidents', 'oseba', 'strokovni_pregled',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'FlightDate',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'DepDelay',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'ArrDelay',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'CarrierDelay',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'WeatherDelay',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'NASDelay',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'SecurityDelay',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'LateAircraftDelay',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'FirstDepTime',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'TotalAddGTime',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'LongestAddGTime',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'DivReachedDest',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'DivActualElapsedTime',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'DivArrDelay',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'DivDistance',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'Div1WheelsOn',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'Div1TotalGTime',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'Div1LongestGTime',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'Div1WheelsOff',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'Div2WheelsOn',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'Div2TotalGTime',),
    ('airline', 'On_Time_On_Time_Performance_2016_1', 'Div2LongestGTime',),
    ('credit', 'charge', 'charge_amt'),
    ('credit', 'member', 'photograph'),
    ('credit', 'member', 'prev_balance'),
    ('credit', 'member', 'curr_balance'),
    ('credit', 'payment', 'payment_amt'),
    ('credit', 'statement', 'statement_amt'),
    ('fhnk', 'zup', 'Pocet'),
    ('fhnk', 'zup', 'Cena'),
    ('financial', 'district', 'A10'),
    ('financial', 'district', 'A12'),
    ('financial', 'district', 'A13'),
    ('financial', 'loan', 'payments'),
    ('financial', 'orders', 'amount'),
    ('geneea', 'bod_schuze', 'uplny_naz'),
    ('geneea', 'funkce', 'nazev_funkce_cz'),
    ('geneea', 'hl_hlasovani', 'cas'),
    ('geneea', 'hl_hlasovani', 'nazev_dlouhy'),
    ('geneea', 'organy', 'nazev_organu_cz'),
    ('geneea', 'organy', 'nazev_organu_en'),
    ('seznam', 'dobito', 'kc_dobito'),
    ('seznam', 'probehnuto', 'kc_proklikano'),
    ('tpc_h', 'part', 'p_retailprice'),
    ('tpc_h', 'supplier', 's_acctbal'),
    ('tpc_h', 'partsupp', 'ps_supplycost'),
    ('tpc_h', 'customer', 'c_acctbal'),
    ('tpc_h', 'orders', 'o_totalprice'),
    ('tpc_h', 'orders', 'o_orderdate'),
    ('tpc_h', 'lineitem', 'l_quantity'),
    ('tpc_h', 'lineitem', 'l_extendedprice'),
    ('tpc_h', 'lineitem', 'l_discount'),
    ('tpc_h', 'lineitem', 'l_tax'),
    ('tpc_h', 'lineitem', 'l_shipdate'),
    ('tpc_h', 'lineitem', 'l_commitdate'),
    ('tpc_h', 'lineitem', 'l_receiptdate'),
)]

is_pk_list = [
    ('db', 'table', 'column'),
    ('accidents', 'nesreca', 'id_nesreca'),
    ('accidents', 'upravna_enota', 'id_upravna_enota'),
    ('airline', 'L_AIRLINE_ID', 'Code'),
    ('airline', 'L_AIRPORT', 'Code'),
    ('airline', 'L_AIRPORT_ID', 'Code'),
    ('airline', 'L_AIRPORT_SEQ_ID', 'Code'),
    ('airline', 'L_CANCELLATION', 'Code'),
    ('airline', 'L_CITY_MARKET_ID', 'Code'),
    ('airline', 'L_DEPARRBLK', 'Code'),
    ('airline', 'L_DISTANCE_GROUP_250', 'Code'),
    ('airline', 'L_DIVERSIONS', 'Code'),
    ('airline', 'L_MONTHS', 'Code'),
    ('airline', 'L_ONTIME_DELAY_GROUPS', 'Code'),
    ('airline', 'L_QUARTERS', 'Code'),
    ('airline', 'L_STATE_ABR_AVIATION', 'Code'),
    ('airline', 'L_STATE_FIPS', 'Code'),
    ('airline', 'L_UNIQUE_CARRIERS', 'Code'),
    ('airline', 'L_WEEKDAYS', 'Code'),
    ('airline', 'L_WORLD_AREA_CODES', 'Code'),
    ('airline', 'L_YESNO_RESP', 'Code'),
    ('baseball', 'allstarfull', 'playerID'),
    ('baseball', 'allstarfull', 'yearID'),
    ('baseball', 'allstarfull', 'gameNum'),
    ('baseball', 'appearances', 'yearID'),
    ('baseball', 'appearances', 'teamID'),
    ('baseball', 'appearances', 'playerID'),
    ('baseball', 'awardsmanagers', 'managerID'),
    ('baseball', 'awardsmanagers', 'awardID'),
    ('baseball', 'awardsmanagers', 'yearID'),
    ('baseball', 'awardsmanagers', 'lgID'),
    ('baseball', 'awardsplayers', 'playerID'),
    ('baseball', 'awardsplayers', 'awardID'),
    ('baseball', 'awardsplayers', 'yearID'),
    ('baseball', 'awardsplayers', 'lgID'),
    ('baseball', 'awardssharemanagers', 'awardID'),
    ('baseball', 'awardssharemanagers', 'yearID'),
    ('baseball', 'awardssharemanagers', 'lgID'),
    ('baseball', 'awardssharemanagers', 'managerID'),
    ('baseball', 'awardsshareplayers', 'awardID'),
    ('baseball', 'awardsshareplayers', 'yearID'),
    ('baseball', 'awardsshareplayers', 'lgID'),
    ('baseball', 'awardsshareplayers', 'playerID'),
    ('baseball', 'batting', 'playerID'),
    ('baseball', 'batting', 'yearID'),
    ('baseball', 'batting', 'stint'),
    ('baseball', 'battingpost', 'yearID'),
    ('baseball', 'battingpost', 'round'),
    ('baseball', 'battingpost', 'playerID'),
    ('baseball', 'fielding', 'playerID'),
    ('baseball', 'fielding', 'yearID'),
    ('baseball', 'fielding', 'stint'),
    ('baseball', 'fielding', 'POS'),
    ('baseball', 'fieldingof', 'playerID'),
    ('baseball', 'fieldingof', 'yearID'),
    ('baseball', 'fieldingof', 'stint'),
    ('baseball', 'fieldingpost', 'playerID'),
    ('baseball', 'fieldingpost', 'yearID'),
    ('baseball', 'fieldingpost', 'round'),
    ('baseball', 'halloffame', 'hofID'),
    ('baseball', 'halloffame', 'yearID'),
    ('baseball', 'managers', 'yearID'),
    ('baseball', 'managers', 'teamID'),
    ('baseball', 'managers', 'inseason'),
    ('baseball', 'managershalf', 'managerID'),
    ('baseball', 'managershalf', 'yearID'),
    ('baseball', 'managershalf', 'teamID'),
    ('baseball', 'managershalf', 'half'),
    ('baseball', 'pitching', 'playerID'),
    ('baseball', 'pitching', 'yearID'),
    ('baseball', 'pitching', 'stint'),
    ('baseball', 'pitchingpost', 'playerID'),
    ('baseball', 'pitchingpost', 'yearID'),
    ('baseball', 'pitchingpost', 'round'),
    ('baseball', 'players', 'lahmanID'),
    ('baseball', 'salaries', 'yearID'),
    ('baseball', 'salaries', 'teamID'),
    ('baseball', 'salaries', 'lgID'),
    ('baseball', 'salaries', 'playerID'),
    ('baseball', 'schools', 'schoolID'),
    ('baseball', 'schoolsplayers', 'playerID'),
    ('baseball', 'schoolsplayers', 'schoolID'),
    ('baseball', 'seriespost', 'yearID'),
    ('baseball', 'seriespost', 'round'),
    ('baseball', 'teams', 'yearID'),
    ('baseball', 'teams', 'lgID'),
    ('baseball', 'teams', 'teamID'),
    ('baseball', 'teamsfranchises', 'franchID'),
    ('baseball', 'teamshalf', 'yearID'),
    ('baseball', 'teamshalf', 'lgID'),
    ('baseball', 'teamshalf', 'teamID'),
    ('baseball', 'teamshalf', 'Half'),
    ('basketball', 'awards_coaches', 'id'),
    ('basketball', 'awards_players', 'playerID'),
    ('basketball', 'awards_players', 'award'),
    ('basketball', 'awards_players', 'year'),
    ('basketball', 'coaches', 'coachID'),
    ('basketball', 'coaches', 'year'),
    ('basketball', 'coaches', 'tmID'),
    ('basketball', 'coaches', 'stint'),
    ('basketball', 'draft', 'id'),
    ('basketball', 'player_allstar', 'playerID'),
    ('basketball', 'player_allstar', 'season_id'),
    ('basketball', 'players', 'playerID'),
    ('basketball', 'players_teams', 'id'),
    ('basketball', 'series_post', 'id'),
    ('basketball', 'teams', 'year'),
    ('basketball', 'teams', 'tmID'),
    ('carcinogenesis', 'atom', 'atomid'),
    ('carcinogenesis', 'canc', 'drug_id'),
    ('carcinogenesis', 'sbond_1', 'id'),
    ('carcinogenesis', 'sbond_2', 'id'),
    ('carcinogenesis', 'sbond_3', 'id'),
    ('carcinogenesis', 'sbond_7', 'id'),
    ('consumer', 'EXPENDITURES', 'EXPENDITURE_ID'),
    ('consumer', 'HOUSEHOLDS', 'HOUSEHOLD_ID'),
    ('credit', 'category', 'category_no'),
    ('credit', 'charge', 'charge_no'),
    ('credit', 'corporation', 'corp_no'),
    ('credit', 'member', 'member_no'),
    ('credit', 'payment', 'payment_no'),
    ('credit', 'provider', 'provider_no'),
    ('credit', 'region', 'region_no'),
    ('credit', 'statement', 'statement_no'),
    ('credit', 'status', 'status_code'),
    ('employee', 'departments', 'dept_no'),
    ('employee', 'dept_emp', 'emp_no'),
    ('employee', 'dept_emp', 'dept_no'),
    ('employee', 'dept_manager', 'dept_no'),
    ('employee', 'dept_manager', 'emp_no'),
    ('employee', 'employees', 'emp_no'),
    ('employee', 'salaries', 'emp_no'),
    ('employee', 'salaries', 'salary'),
    ('employee', 'titles', 'emp_no'),
    ('employee', 'titles', 'title'),
    ('employee', 'titles', 'from_date'),
    ('fhnk', 'pripady', 'Identifikace_pripadu'),
    ('fhnk', 'vykony', 'Identifikace_pripadu'),
    ('fhnk', 'vykony', 'Datum_provedeni_vykonu'),
    ('fhnk', 'vykony', 'Kod_polozky'),
    ('fhnk', 'zup', 'Identifikace_pripadu'),
    ('fhnk', 'zup', 'Datum_provedeni_vykonu'),
    ('fhnk', 'zup', 'Kod_polozky'),
    ('financial', 'account', 'account_id'),
    ('financial', 'card', 'card_id'),
    ('financial', 'client', 'client_id'),
    ('financial', 'disp', 'disp_id'),
    ('financial', 'district', 'district_id'),
    ('financial', 'loan', 'loan_id'),
    ('financial', 'orders', 'order_id'),
    ('financial', 'trans', 'trans_id'),
    ('geneea', 'bod_stav', 'id_bod_stav'),
    ('geneea', 'funkce', 'id_funkce'),
    ('geneea', 'hl_hlasovani', 'id_hlasovani'),
    ('geneea', 'organy', 'id_organ'),
    ('geneea', 'osoby', 'id_osoba'),
    ('geneea', 'poslanec', 'id_poslanec'),
    ('geneea', 'schuze', 'id_schuze'),
    ('geneea', 'typ_funkce', 'id_typ_funkce'),
    ('geneea', 'typ_organu', 'id_typ_org'),
    ('genome', 'ATT_CLASSES', 'ATT_CLASS_ID'),
    ('genome', 'IMG_OBJ', 'IMG_ID'),
    ('genome', 'IMG_OBJ', 'OBJ_SAMPLE_ID'),
    ('genome', 'IMG_OBJ_ATT', 'IMG_ID'),
    ('genome', 'IMG_OBJ_ATT', 'ATT_CLASS_ID'),
    ('genome', 'IMG_OBJ_ATT', 'OBJ_SAMPLE_ID'),
    ('genome', 'IMG_REL', 'IMG_ID'),
    ('genome', 'IMG_REL', 'PRED_CLASS_ID'),
    ('genome', 'IMG_REL', 'OBJ1_SAMPLE_ID'),
    ('genome', 'IMG_REL', 'OBJ2_SAMPLE_ID'),
    ('genome', 'OBJ_CLASSES', 'OBJ_CLASS_ID'),
    ('genome', 'OBJ_CLASSES', 'OBJ_CLASS'),
    ('genome', 'PRED_CLASSES', 'PRED_CLASS_ID'),
    ('hepatitis', 'Bio', 'b_id'),
    ('hepatitis', 'dispat', 'm_id'),
    ('hepatitis', 'indis', 'in_id'),
    ('hepatitis', 'inf', 'a_id'),
    ('hepatitis', 'rel11', 'b_id'),
    ('hepatitis', 'rel11', 'm_id'),
    ('hepatitis', 'rel12', 'in_id'),
    ('hepatitis', 'rel12', 'm_id'),
    ('hepatitis', 'rel13', 'a_id'),
    ('hepatitis', 'rel13', 'm_id'),
    ('imdb', 'aka_name', 'id'),
    ('imdb', 'aka_title', 'id'),
    ('imdb', 'cast_info', 'id'),
    ('imdb', 'char_name', 'id'),
    ('imdb', 'comp_cast_type', 'id'),
    ('imdb', 'company_name', 'id'),
    ('imdb', 'company_type', 'id'),
    ('imdb', 'complete_cast', 'id'),
    ('imdb', 'info_type', 'id'),
    ('imdb', 'keyword', 'id'),
    ('imdb', 'kind_type', 'id'),
    ('imdb', 'link_type', 'id'),
    ('imdb', 'movie_companies', 'id'),
    ('imdb', 'movie_info_idx', 'id'),
    ('imdb', 'movie_keyword', 'id'),
    ('imdb', 'movie_link', 'id'),
    ('imdb', 'name', 'id'),
    ('imdb', 'role_type', 'id'),
    ('imdb', 'title', 'id'),
    ('imdb', 'movie_info', 'id'),
    ('imdb', 'person_info', 'id'),
    ('movielens', 'actors', 'actorid'),
    ('movielens', 'directors', 'directorid'),
    ('movielens', 'movies', 'movieid'),
    ('movielens', 'movies2actors', 'movieid'),
    ('movielens', 'movies2actors', 'actorid'),
    ('movielens', 'movies2directors', 'movieid'),
    ('movielens', 'movies2directors', 'directorid'),
    ('movielens', 'u2base', 'userid'),
    ('movielens', 'u2base', 'movieid'),
    ('movielens', 'users', 'userid'),
    ('seznam', 'client', 'client_id'),
    ('seznam', 'probehnuto_mimo_penezenku', 'client_id'),
    ('seznam', 'probehnuto_mimo_penezenku', 'Month_Year'),
    ('ssb', 'customer', 'c_custkey'),
    ('ssb', 'part', 'p_partkey'),
    ('ssb', 'supplier', 's_suppkey'),
    ('ssb', 'dim_date', 'd_datekey'),
    ('tournament', 'regular_season_compact_results', 'season'),
    ('tournament', 'regular_season_compact_results', 'daynum'),
    ('tournament', 'regular_season_compact_results', 'wteam'),
    ('tournament', 'regular_season_compact_results', 'lteam'),
    ('tournament', 'regular_season_detailed_results', 'season'),
    ('tournament', 'regular_season_detailed_results', 'daynum'),
    ('tournament', 'regular_season_detailed_results', 'wteam'),
    ('tournament', 'regular_season_detailed_results', 'lteam'),
    ('tournament', 'seasons', 'season'),
    ('tournament', 'target', 'id'),
    ('tournament', 'teams', 'team_id'),
    ('tournament', 'tourney_compact_results', 'season'),
    ('tournament', 'tourney_compact_results', 'daynum'),
    ('tournament', 'tourney_compact_results', 'wteam'),
    ('tournament', 'tourney_compact_results', 'lteam'),
    ('tournament', 'tourney_detailed_results', 'season'),
    ('tournament', 'tourney_detailed_results', 'daynum'),
    ('tournament', 'tourney_detailed_results', 'wteam'),
    ('tournament', 'tourney_detailed_results', 'lteam'),
    ('tournament', 'tourney_seeds', 'season'),
    ('tournament', 'tourney_seeds', 'seed'),
    ('tournament', 'tourney_slots', 'season'),
    ('tournament', 'tourney_slots', 'slot'),
    ('walmart', 'key', 'store_nbr'),
    ('walmart', 'station', 'station_nbr'),
    ('walmart', 'train', 'date'),
    ('walmart', 'train', 'store_nbr'),
    ('walmart', 'train', 'item_nbr'),
]


def get_table_col_names(dbms_wrapper: DBMSWrapper) -> List[Dict[str, Any]]:
    if dbms_wrapper.dbms == 'postgres':
        raise NotImplementedError('Not implemented for postgres')
    elif dbms_wrapper.dbms == 'duckdb':
        stats_query = """
                SELECT table_name, column_name, data_type FROM duckdb_columns;
                """
        column_stats_names = ['table', 'column', 'datatype']
        column_stats_rows = dbms_wrapper.exec(stats_query)
        return [{k: v for k, v in zip(column_stats_names, row)} for row in column_stats_rows]


def get_table_size(dbms_wrapper: DBMSWrapper, table: str):
    count_query = f'SELECT count(*) FROM {table};'

    card = dbms_wrapper.exec(count_query)[0][0]
    assert isinstance(card, int), f'Cardinality is not an integer: {card} / {type(card)}'
    return card


def get_nan_ratio(dbms_wrapper: DBMSWrapper, table: str, column: str, table_card: int):
    nan_query = f'SELECT count(*) FROM {table} WHERE "{column}" IS NULL;'
    card = dbms_wrapper.exec(nan_query)[0][0]
    assert isinstance(card, int), f'Cardinality is an integer: {card} / {type(card)}'

    if table_card == 0:
        assert card == 0, f'Cardinality is not 0: {card} / {type(card)}'
        return 0

    return card / table_card


def create_db_statistics(dbms: str, dbms_kwargs: dict, col_stats_dir: str, target: str, datasets: str):
    statistics_dict = {}

    for dataset in dataset_list_dict[datasets]:
        dbms_wrapper = DBMSWrapper(dbms=dbms, dbms_kwargs=dbms_kwargs, db_name=dataset.db_name)

        dataset_info = get_table_col_names(dbms_wrapper)
        assert len(dataset_info) > 0, f'No info found for dataset {dataset.db_name}'

        # gather table cardinalities
        tables = list(set([t['table'] for t in dataset_info]))
        table_card_dict = {}
        table_num_numeric_cols = {}
        for table in tables:
            table_card_dict[table] = get_table_size(dbms_wrapper, table)

            # gather num numeric columns
            table_num_numeric_cols[table] = len(
                [t for t in dataset_info if
                 t['table'] == table and (
                         t['datatype'].lower() in ['integer', 'float', 'double precision', 'bigint'] or t[
                     'datatype'].startswith('DECIMAL'))])

        # augment percentile information
        col_stats_path = os.path.join(col_stats_dir, f'{dataset.db_name}_column_statistics.json')
        # open the column_statistics.json file to a dictionary
        with open(col_stats_path, "r") as f:
            col_stats = json.load(f)

        # augment dataset_info with cardinalities
        for entry in dataset_info:
            table = entry['table']
            column = entry['column']

            entry['table_rows'] = table_card_dict[table]
            entry['no_num_values'] = table_num_numeric_cols[table]

            # gather nan ratio
            entry['nan_ratio'] = get_nan_ratio(dbms_wrapper, table, column,
                                               entry['table_rows'])

            entry['PK'] = '1' if (dataset.db_name, table, column) in is_pk_list else '0'
            entry['use'] = 'NO' if (dataset.db_name, table, column) in skip_col_list else 'YES'

            if table in col_stats and column in col_stats[table] and 'percentiles' in col_stats[table][column]:
                for i in range(11):
                    entry['percentile_' + str(i * 10)] = col_stats[table][column]["percentiles"][i]
            else:
                for i in range(11):
                    entry['percentile_' + str(i * 10)] = None

        assert len(dataset_info) > 0, f'No columns found for dataset {dataset.db_name}'

        statistics_dict[dataset.db_name] = dataset_info

    # linearize dict
    data_list = []
    for dataset, data in statistics_dict.items():
        for row in data:
            row['db'] = dataset
            data_list.append(row)
    df = pd.DataFrame(data_list,
                      columns=['db', 'table', 'column', 'datatype', 'table_rows', 'nan_ratio', 'PK', 'use',
                               'no_num_values'] + [f'percentile_{i * 10}' for i in range(11)])
    print(f'Num stats entries: {len(df)}')
    df.to_csv(target, index=False)
    print(f'Stats have been written to: {target}')
