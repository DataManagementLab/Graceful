from models.zero_shot_models.zero_shot_model import ZeroShotModel


class PostgresZeroShotModel(ZeroShotModel):
    """
    Zero-shot cost estimation model for postgres.
    """

    def __init__(self, featurization=None, plans_have_no_udf: bool = False, **zero_shot_kwargs):
        # define the MLPs for the different node types in the graph representation of queries
        encoders = [
            ('column', featurization['COLUMN_FEATURES']),
            ('table', featurization['TABLE_FEATURES']),
            ('output_column', featurization['OUTPUT_COLUMN_FEATURES']),
            ('filter_column', featurization['FILTER_FEATURES'] + featurization['COLUMN_FEATURES']),
            ('plan', featurization['PLAN_FEATURES']),
            ('logical_pred', featurization['FILTER_FEATURES']),

        ]

        if not plans_have_no_udf:
            encoders += [
                # Additions for UDF support
                ('INV', featurization['INV_FEATURES']),
                ('COMP', featurization['COMP_FEATURES']),
                ('BRANCH', featurization['BRANCH_FEATURES']),
                ('LOOP', featurization['LOOP_FEATURES']),
                ('LOOPEND', featurization['LOOPEND_FEATURES']),
                ('RET', featurization['RET_FEATURES']),
            ]

        # define messages passing which is peculiar for postgres
        if plans_have_no_udf:
            prepasses = [
                dict(model_name='col_output_col', e_name='col_output_col', allow_empty=True),
            ]
            post_udf_passes = []
        else:
            prepasses = [
                dict(model_name='col_output_col', e_name='col_output_col', allow_empty=True),
                # is empty in case of UDF on select only queries
                dict(model_name='col_COMP', e_name='col_COMP'),
                dict(model_name='col_INV', e_name='col_INV'),
            ]
            post_udf_passes = [
                dict(model_name='RET_outcol', e_name='RET_outcol'),
                dict(model_name='RET_filter', e_name='RET_filter', allow_empty=True),
                # is empty in case of UDF on select only queries
            ]

        tree_model_types = [e['model_name'] for e in prepasses] + [e['model_name'] for e in post_udf_passes]

        super().__init__(featurization=featurization, encoders=encoders, prepasses=prepasses,
                         post_udf_passes=post_udf_passes,
                         add_tree_model_types=tree_model_types, plans_have_no_udf=plans_have_no_udf, **zero_shot_kwargs)
