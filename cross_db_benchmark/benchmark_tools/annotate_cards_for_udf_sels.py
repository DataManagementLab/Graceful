from typing import Dict


def annotate_udf_info_to_plan(parsed_runs: Dict):
    plans = parsed_runs['parsed_plans']
    for plan in plans:
        udf_filter_contained = _check_that_plan_contains_udf_filter(plan)

        # annotate udf selectivities
        _recursively_annotate_udf_pos_info(plan, udf_filter_contained=udf_filter_contained)


def _check_that_plan_contains_udf_filter(plan: Dict) -> bool:
    """
    Check if the plan contains UDF filter node
    """
    if not 'udf' in plan:
        # no udf in the plan
        return False

    if plan['udf']['udf_pos_in_query'] == 'filter':
        return True
    else:
        return False


def _recursively_annotate_udf_pos_info(plan: Dict, udf_filter_contained: bool):
    """
    Recursively apply UDF filter selectivity to the plan from top until the udf filter node. Beyond still add the keywords but copy the value
    """

    this_is_udf = False
    if 'filter_columns' in plan['plan_parameters'] and hasattr(plan['plan_parameters']['filter_columns'],
                                                               'udf_name') and plan['plan_parameters'][
        'filter_columns'].udf_name is not None:
        # stop applying the selectivity after the udf filter node
        this_is_udf = True

    if this_is_udf:
        udf_filter_contained = False

    plan['plan_parameters']['above_udf_filter'] = udf_filter_contained
    plan['plan_parameters']['is_udf_filter'] = this_is_udf

    for child in plan['children']:
        _recursively_annotate_udf_pos_info(child, udf_filter_contained=udf_filter_contained)
