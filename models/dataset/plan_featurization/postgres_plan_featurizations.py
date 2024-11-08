# different features used for plan nodes, filter columns etc. used for postgres plans

PostgresUDFTest = dict(
    PLAN_FEATURES=['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['reltuples', 'relpages'],
    INVOC_FEATURES=['in_rows', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows', 'lib', 'ops', 'loop_part'],
    RETURN_FEATURES=['out_dts', 'in_rows'],
    BRANCH_FEATURES=['in_rows', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

DuckDBActUDF = dict(
    PLAN_FEATURES=['act_card', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_act', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_act', 'ops', 'loop_part'],
    RET_FEATURES=['out_dts', 'in_rows_act'],
    BRANCH_FEATURES=['in_rows_act', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
    LOOPEND_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

DuckDBActUDFFilterUDF = dict(
    PLAN_FEATURES=['act_card', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_act', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_act', 'ops', 'loop_part'],
    RET_FEATURES=['out_dts', 'in_rows_act'],
    BRANCH_FEATURES=['in_rows_act', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
    LOOPEND_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

DuckDBEstUDF = dict(
    PLAN_FEATURES=['est_card', 'op_name', 'est_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_est', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_est', 'ops', 'loop_part'],
    RET_FEATURES=['out_dts', 'in_rows_est'],
    BRANCH_FEATURES=['in_rows_est', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows_est', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
    LOOPEND_FEATURES=['in_rows_est', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

DuckDBEstUDFFilterUDF = dict(
    PLAN_FEATURES=['est_card', 'op_name', 'est_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_est', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_est', 'ops', 'loop_part'],
    RET_FEATURES=['out_dts', 'in_rows_est'],
    BRANCH_FEATURES=['in_rows_est', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows_est', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
    LOOPEND_FEATURES=['in_rows_est', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

DuckDBDeepUDFFilterUDF = dict(
    PLAN_FEATURES=['dd_est_card', 'op_name', 'dd_est_children_card', 'above_udf_filter'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_deepdb', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_deepdb', 'ops', 'loop_part'],
    RET_FEATURES=['out_dts', 'in_rows_deepdb'],
    BRANCH_FEATURES=['in_rows_deepdb', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows_deepdb', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
    LOOPEND_FEATURES=['in_rows_deepdb', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

DuckDBWJUDFFilterUDF = dict(
    PLAN_FEATURES=['wj_est_card', 'op_name', 'wj_est_children_card', 'above_udf_filter'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_deepdb', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_deepdb', 'ops', 'loop_part'],
    RET_FEATURES=['out_dts', 'in_rows_deepdb'],
    BRANCH_FEATURES=['in_rows_deepdb', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows_deepdb', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
    LOOPEND_FEATURES=['in_rows_deepdb', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

PostgresTrueCardDetail = dict(
    PLAN_FEATURES=['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac'],
    OUTPUT_COLUMN_FEATURES=['aggregation'],
    TABLE_FEATURES=['reltuples', 'relpages'],
)

PostgresEstSystemCardDetail = dict(
    PLAN_FEATURES=['est_card', 'est_width', 'workers_planned', 'op_name', 'est_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac'],
    OUTPUT_COLUMN_FEATURES=['aggregation'],
    TABLE_FEATURES=['reltuples', 'relpages'],
)

PostgresDeepDBEstSystemCardDetail = dict(
    PLAN_FEATURES=['dd_est_card', 'est_width', 'workers_planned', 'op_name', 'dd_est_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac'],
    OUTPUT_COLUMN_FEATURES=['aggregation'],
    TABLE_FEATURES=['reltuples', 'relpages'],
)

featurization_dict = {
    'ddact': DuckDBActUDF,
    'ddactfonudf': DuckDBActUDFFilterUDF,
    'ddest': DuckDBEstUDF,
    'ddestfonudf': DuckDBEstUDFFilterUDF,
    'dddeepfonudf': DuckDBDeepUDFFilterUDF,
    'ddwjfonudf': DuckDBWJUDFFilterUDF,
    'pgzsdd': PostgresDeepDBEstSystemCardDetail,
    'pgzsact': PostgresTrueCardDetail,
}
