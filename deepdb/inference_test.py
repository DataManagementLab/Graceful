import time

import deepdb

if __name__ == '__main__':
    print(deepdb.__version__)
    db_name = 'basketball_scaled200'
    db_dir_name = 'basketball_scaled200'
    ensemble_location = '../data/ensemble_relationships_basketball_scaled200_0.3_10000000.pkl'
    from deepdb.inference import FilterCond, DeepDBEstimator
    t0 = time.time()
    deepdb_estimator = DeepDBEstimator(ensemble_location=ensemble_location, db_name=db_name, scale=1)
    print(f"Time to load deepdb estimator: {time.time() - t0}")

    filter_conditions = [
        FilterCond(table_name='teams', column_name='min', operator='<=', value=19878.092123331367),
        FilterCond(table_name='teams', column_name='d_tmRebound', operator='>=', value=0),
        FilterCond(table_name='teams', column_name='o_stl', operator='<=', value=632),
        FilterCond(table_name='series_post', column_name='L', operator='!=', value=2),
    ]

    result = deepdb_estimator.estimate_card(filter_conditions=frozenset(filter_conditions),
                                            tables=frozenset(['teams', 'series_post']),
                                            join_conditions=tuple(['teams.tmID = series_post.tmIDWinner']))
    print(result)
