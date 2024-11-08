import math
from collections import defaultdict

import numpy as np
import wandb

from models.training.metrics import QError


def _slice_and_log(x: np.ndarray, y: np.ndarray, slice_by: np.ndarray, slice_by_title: str, log_to_wandb: bool = True,
                   prefix: str = ''):
    assert len(x) == len(y) == len(
        slice_by), f"lengths of x, y, and slice_by must be equal, but are {len(x)}, {len(y)}, {len(slice_by)} ({slice_by_title})"

    x_slices_dict = dict()
    y_slices_dict = dict()

    # Identify the unique values in the slice_by array
    unique_slicing_vals = np.unique(slice_by)
    try:
        max_slicing_val = math.ceil(max(unique_slicing_vals))
    except Exception as e:
        print(e)
        max_slicing_val = 0

    if len(unique_slicing_vals) > 10 and (
            isinstance(unique_slicing_vals[0], np.int64) or isinstance(unique_slicing_vals[0], np.float64)):
        # too many unique values, bin values
        bin_edges = np.linspace(0, max_slicing_val, 10)
        slice_by_bin_id = np.digitize(slice_by, bin_edges, right=True)
        slice_by = bin_edges[slice_by_bin_id]
        unique_slicing_vals = np.unique(slice_by)

    # for each unique value, create a list of tuples (x,y) that have that value
    for value in unique_slicing_vals:
        bool_arr = slice_by == value
        x_slices_dict[value] = x[bool_arr]
        y_slices_dict[value] = y[bool_arr]

    # compute the qerror for each slice
    metrics = [QError(percentile=50, verbose=False), QError(percentile=95, verbose=False),
               QError(percentile=100, verbose=False)]

    # run the metrics on each slice and log the results
    qerror_results = defaultdict(dict)

    for val in unique_slicing_vals:
        y_slice = y_slices_dict[val]
        x_slice = x_slices_dict[val]

        if isinstance(val, str):
            val_str = val
        elif isinstance(val, float) or isinstance(val, np.float64):
            val = float(val)
            val_str = f'{val:.2f}'
            val_str = val_str.zfill(len(f'{max_slicing_val:.2f}'))
        elif isinstance(val, int) or isinstance(val, np.int64):
            val_str = str(val).zfill(len(str(max_slicing_val)))
        else:
            raise ValueError(f'Unknown type {type(val)} for {val}')

        slice_str = f'{val_str} ({len(x_slice)} entries)'

        for metric in metrics:
            try:
                error = metric.evaluate_metric(y_slice, x_slice)
            except Exception as e:
                print(
                    f'Error evaluating metric {metric.name} by {slice_by_title} for slice {val_str} ({len(x_slice)} entries)')
                raise e
            qerror_results[metric.name][slice_str] = error

    # log the results
    if log_to_wandb:
        table_data = []
        for slice_val in sorted(qerror_results[metrics[0].name].keys()):
            table_data.append([slice_val] + [qerror_results[metric.name][slice_val] for metric in metrics])
        columns = [slice_by_title] + [metric.name for metric in metrics]

        table = wandb.Table(
            data=table_data,
            columns=columns)

        for metric_name, metric_results in qerror_results.items():
            wandb.log({f"{prefix}{metric_name} by {slice_by_title}": wandb.plot.bar(table, slice_by_title, metric_name,
                                                                                    title=f"{metric_name} by {slice_by_title}")})

    return qerror_results


def slice_evaluation_output(predictions: np.ndarray, labels: np.ndarray, num_joins: np.ndarray,
                            num_filters: np.ndarray,
                            udf_num_np_calls: np.ndarray, udf_num_math_calls: np.ndarray,
                            udf_num_comp_nodes: np.ndarray,
                            udf_num_branches: np.ndarray, udf_num_loops: np.ndarray, udf_pos_in_query: np.ndarray,
                            udf_in_card: np.ndarray, udf_filter_num_logicals: np.ndarray,
                            udf_filter_num_literals: np.ndarray,
                            prefix: str,
                            log_to_wandb: bool = True):
    results_json = dict()

    # slice by label
    bin_edges_s = np.asarray([0, 0.5, 1, 1.5, 2, 5, 10, 30, 1000])
    results_json[f'{prefix}_slice_by_labels'] = _slice_and_log(predictions, labels,
                                                               bin_edges_s[
                                                                   np.digitize(labels, bin_edges_s, right=True)],
                                                               "runtime", log_to_wandb=log_to_wandb, prefix=prefix)

    # slice by num_joins
    results_json[f'{prefix}_slice_by_joins'] = _slice_and_log(predictions, labels, num_joins, "num_joins",
                                                              log_to_wandb=log_to_wandb, prefix=prefix)

    # slice by num_filters
    results_json[f'{prefix}_slice_by_filters'] = _slice_and_log(predictions, labels, num_filters, "num_filters",
                                                                log_to_wandb=log_to_wandb, prefix=prefix)

    # slice by num_joins and num_filters
    num_joins_filters = np.array([f'{num_joins[i]}_{num_filters[i]}' for i in range(len(num_joins))])
    results_json[f'{prefix}_slice_by_joins_filters'] = _slice_and_log(predictions, labels, num_joins_filters,
                                                                      "num_join_filters", log_to_wandb=log_to_wandb,
                                                                      prefix=prefix)

    # slice by udf_num_np_calls
    results_json[f'{prefix}_slice_by_udf_num_np_calls'] = _slice_and_log(predictions, labels, udf_num_np_calls,
                                                                         "udf num np calls", log_to_wandb=log_to_wandb,
                                                                         prefix=prefix)

    # slice by udf_num_math_calls
    results_json[f'{prefix}_slice_by_udf_num_math_calls'] = _slice_and_log(predictions, labels, udf_num_math_calls,
                                                                           "udf num math calls",
                                                                           log_to_wandb=log_to_wandb, prefix=prefix)

    # slice by udf_num_nodes
    results_json[f'{prefix}_slice_by_udf_num_comp_nodes'] = _slice_and_log(predictions, labels, udf_num_comp_nodes,
                                                                           "udf num comp nodes",
                                                                           log_to_wandb=log_to_wandb, prefix=prefix)

    # slice by udf_num_branches
    results_json[f'{prefix}_slice_by_udf_num_branches'] = _slice_and_log(predictions, labels, udf_num_branches,
                                                                         "udf num branches", log_to_wandb=log_to_wandb,
                                                                         prefix=prefix)

    # slice by udf_num_loops
    results_json[f'{prefix}_slice_by_udf_num_loops'] = _slice_and_log(predictions, labels, udf_num_loops,
                                                                      "udf num loops", log_to_wandb=log_to_wandb,
                                                                      prefix=prefix)

    # slice by pos in query
    translate_dict = {'filter': 0, 'filter_pullup': 1, 'select': 2, 'none': 3}
    udf_pos_in_query = np.array([translate_dict[val] for val in udf_pos_in_query])

    results_json[f'{prefix}_slice_by_udf_pos_in_query'] = _slice_and_log(predictions, labels, udf_pos_in_query,
                                                                         "udf pos in query (0=filter (pushdown), 1 filter (pullup), 2=select)",
                                                                         log_to_wandb=log_to_wandb, prefix=prefix)

    # slice by in card
    results_json[f'{prefix}_slice_by_udf_in_card'] = _slice_and_log(predictions, labels, udf_in_card,
                                                                    "udf in-card",
                                                                    log_to_wandb=log_to_wandb, prefix=prefix)

    # slice by udf_filter_num_logicals
    results_json[f'{prefix}_slice_by_udf_filter_num_logicals'] = _slice_and_log(predictions, labels,
                                                                                udf_filter_num_logicals,
                                                                                "udf filter num logicals",
                                                                                log_to_wandb=log_to_wandb,
                                                                                prefix=prefix)

    # slice by udf_filter_num_literals
    results_json[f'{prefix}_slice_by_udf_filter_num_literals'] = _slice_and_log(predictions, labels,
                                                                                udf_filter_num_literals,
                                                                                "udf filter num literals",
                                                                                log_to_wandb=log_to_wandb,
                                                                                prefix=prefix)

    return results_json
