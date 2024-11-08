import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from models.dataset.dataset_creation import read_workload_runs, create_datasets
from models.training.metrics import QError
from utils.hyperparams_utils import config_keywords


def create_pull_push_label_plot(pullup_labels, pushdown_labels, dataset: str = None):
    # create statistic about pullup / pushdown runtime distribution
    import matplotlib.pyplot as plt

    tmp_pull = pullup_labels[:]
    tmp_push = pushdown_labels[:]

    # sort both lists by pushdown_labels
    tmp_pull = [x for _, x in sorted(zip(tmp_push, tmp_pull), key=lambda pair: pair[0])]
    tmp_push = sorted(tmp_push)

    # create barplot with both lists
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    barWidth = 0.4
    r1 = range(len(tmp_pull))
    r2 = [x + barWidth for x in r1]
    ax.bar(r1, tmp_pull, color='b', width=barWidth, label='Pullup')
    ax.bar(r2, tmp_push, color='r', width=barWidth, label='Pushdown')
    ax.set_xlabel(f'Query (id) - {len(tmp_pull)} queries')
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f'Runtime of Pullup and Pushdown Plans ({dataset})')
    ax.set_xticks([r + barWidth / 2 for r in range(len(tmp_pull))])
    ax.set_xticklabels(range(1, len(tmp_pull) + 1))
    ax.legend()

    print(
        f'Max possible acceleration: {(sum(tmp_push) - sum([min(l1, l2) for l1, l2 in zip(tmp_pull, tmp_push)])) / sum(tmp_push) * 100:.2f}%')
    return plt


def convert_dict_of_lists_to_dataframe(dict_of_lists):
    import pandas as pd

    # translate keys to strings
    new_dict = {}
    for key, value in dict_of_lists.items():
        new_dict[str(key)] = value

    return pd.DataFrame(new_dict)


def assemble_pullup_pushdown_overlap_datasets(pullup_plans_path: str, pushdown_plans_path: str, min_runtime_ms: int = 0,
                                              max_runtime: int = 30):
    plans_pushdown, pushdown_dataset_stats = read_workload_runs([pushdown_plans_path], min_runtime_ms=min_runtime_ms,
                                                                max_runtime=max_runtime)
    plans_pullup, pullup_dataset_stats = read_workload_runs([pullup_plans_path], min_runtime_ms=min_runtime_ms,
                                                            max_runtime=max_runtime)

    # load col_col comp stats
    dataset = os.path.basename(os.path.dirname(pullup_plans_path))
    exp_path = os.path.dirname(os.path.dirname(os.path.dirname(pullup_plans_path)))
    col_col_stats_path = os.path.join(exp_path, 'dbs', dataset, 'created_graphs', 'udf_w_col_col_comparison.json')
    with open(col_col_stats_path, 'r') as f:
        fn_blacklist = json.load(f)

    # create dict udf func to plans
    udf_plans_pushdown_dict = {}
    udf_plans_pullup_dict = {}

    for plan in plans_pushdown:
        udf_plans_pushdown_dict[plan.udf.udf_name] = plan

    for plan in plans_pullup:
        udf_plans_pullup_dict[plan.udf.udf_name] = plan

    # filter out blacklisted udfs
    udf_plans_pushdown_dict = {k: v for k, v in udf_plans_pushdown_dict.items() if k not in fn_blacklist}
    udf_plans_pullup_dict = {k: v for k, v in udf_plans_pullup_dict.items() if k not in fn_blacklist}

    # check for overlap in dicts
    overlap = list(set(udf_plans_pushdown_dict.keys()).intersection(set(udf_plans_pullup_dict.keys())))

    # filter out plans which differ in runtime (pullup vs. pushdown) by <5% of the runtime
    # this is to make the benchmark harder because plans where we don't care about the result are excluded
    filtered_overlap = []
    for udf_name in overlap:
        pushdown_runtime = udf_plans_pushdown_dict[udf_name].plan_runtime_ms
        pullup_runtime = udf_plans_pullup_dict[udf_name].plan_runtime_ms

        if max(pushdown_runtime, pullup_runtime) / min(pushdown_runtime, pullup_runtime) > 1.05:
            filtered_overlap.append(udf_name)

    print(
        f'Filtered out {len(overlap) - len(filtered_overlap)} UDFs because runtime difference between pushdown and pullup plan is negligible. {len(filtered_overlap)} UDFs remaining.')

    # make sure overlap is no longer used
    del overlap

    # extract overlapping plans
    overlap_plans_pushdown = [udf_plans_pushdown_dict[udf_name] for udf_name in filtered_overlap]
    overlap_plans_pullup = [udf_plans_pullup_dict[udf_name] for udf_name in filtered_overlap]
    pushdown_sql = [plan.query for plan in overlap_plans_pushdown]
    pullup_sql = [plan.query for plan in overlap_plans_pullup]

    _, pullup_test_dataset, _, _, _, pullup_test_database_statistics = create_datasets(None,
                                                                                       loss_class_name=None,
                                                                                       val_ratio=0,
                                                                                       shuffle_before_split=False,
                                                                                       stratify_dataset_by_runtimes=False,
                                                                                       # avoid double stratification. Perform only once since we only have one database anyways here
                                                                                       max_runtime=max_runtime,
                                                                                       zs_paper_dataset=False,
                                                                                       train_udf_graph_against_udf_runtime=False,
                                                                                       min_runtime_ms=min_runtime_ms,
                                                                                       infuse_plans=overlap_plans_pullup,
                                                                                       infuse_database_statistics=pullup_dataset_stats)

    _, pushdown_test_dataset, _, _, _, pushdown_test_database_statistics = create_datasets(None,
                                                                                           loss_class_name=None,
                                                                                           val_ratio=0,
                                                                                           shuffle_before_split=False,
                                                                                           stratify_dataset_by_runtimes=False,
                                                                                           # avoid double stratification. Perform only once since we only have one database anyways here
                                                                                           max_runtime=max_runtime,
                                                                                           zs_paper_dataset=False,
                                                                                           train_udf_graph_against_udf_runtime=False,
                                                                                           min_runtime_ms=min_runtime_ms,
                                                                                           infuse_plans=overlap_plans_pushdown,
                                                                                           infuse_database_statistics=pushdown_dataset_stats)

    return pullup_test_dataset, pullup_test_database_statistics, pushdown_test_dataset, pushdown_test_database_statistics, pushdown_sql, pullup_sql


def log_q_errors(preds, labels):
    qerrors = [
        ('q50', QError(percentile=50, verbose=False)),
        ('q95', QError(percentile=95, verbose=False)),
        ('q100', QError(percentile=100, verbose=False))
    ]
    d = dict()
    for m_name, metric in qerrors:
        value = metric.evaluate_metric(labels, preds)
        d[m_name] = value
        print(f'{m_name} : {value}')
    return d


def extract_train_card_type(orig_args_config: Dict):
    train_featurization_keyword = None
    for config_keyword, type in config_keywords.items():
        if type == 'plan_featurization' and f'{config_keyword}_' in orig_args_config['model_config']:
            train_featurization_keyword = config_keyword
            break

    assert train_featurization_keyword is not None

    # extract which cardinalities have been used during training (e.g. act / est / deep)
    if 'act' in train_featurization_keyword:
        train_card_type = 'act_card'
    elif 'est' in train_featurization_keyword:
        train_card_type = 'est_card'
    elif 'deep' in train_featurization_keyword:
        train_card_type = 'dd_est_card'
    elif 'wj' in train_featurization_keyword:
        train_card_type = 'wj_est_card'
    else:
        raise ValueError(
            f'Could not extract cardinality estimation method from train_featurization_keyword: {train_featurization_keyword}')
    return train_card_type


def gen_qerror_plot(pushdown_qerrors, pullup_qerrors):
    plt.close()
    instance_names = []
    instance_specs = []
    pullup_q50_errors = []
    pullup_q95_errors = []
    pullup_q100_errors = []

    for card, value in pullup_qerrors.items():
        for sel, errors in value.items():
            if sel == 'None':
                instance_names.append(f'{card}')
            else:
                instance_names.append(f'{card} - {sel}')
            instance_specs.append((card, sel))
            pullup_q50_errors.append(errors['q50'])
            pullup_q95_errors.append(errors['q95'])
            pullup_q100_errors.append(errors['q100'])

    pushdown_q50_errors = []
    pushdown_q95_errors = []
    pushdown_q100_errors = []

    for card, sel in instance_specs:
        if card not in pushdown_qerrors:
            pushdown_q50_errors.append(np.nan)
            pushdown_q95_errors.append(np.nan)
            pushdown_q100_errors.append(np.nan)
        else:
            pushdown_q50_errors.append(pushdown_qerrors[card][sel]['q50'])
            pushdown_q95_errors.append(pushdown_qerrors[card][sel]['q95'])
            pushdown_q100_errors.append(pushdown_qerrors[card][sel]['q100'])

    assert len(pullup_q50_errors) == len(pushdown_q50_errors) == len(pullup_q95_errors) == len(
        pushdown_q95_errors) == len(pullup_q100_errors) == len(pushdown_q100_errors) == len(instance_names)

    # create barplot with both lists
    fig, axs = plt.subplots(3, 2, figsize=(10, 20))

    def plot_to_ax(ax, data, title, max):
        ax.bar(instance_names, data)
        ax.set_ylim(1, max * 1.1)
        ax.set_title(title)
        ax.set_ylabel('Runtime Est QError')
        ax.set_xlabel('Instance')
        ax.set_xticklabels(instance_names, rotation=45, ha='right')

    r0_max = max(max(pullup_q50_errors), max(pushdown_q50_errors))
    r1_max = max(max(pullup_q95_errors), max(pushdown_q95_errors))
    r2_max = max(max(pullup_q100_errors), max(pushdown_q100_errors))

    plot_to_ax(axs[0, 0], pullup_q50_errors, 'Pullup q50 errors', max=r0_max)
    plot_to_ax(axs[1, 0], pullup_q95_errors, 'Pullup q95 errors', max=r1_max)
    plot_to_ax(axs[2, 0], pullup_q100_errors, 'Pullup q100 errors', max=r2_max)
    plot_to_ax(axs[0, 1], pushdown_q50_errors, 'Pushdown q50 errors', max=r0_max)
    plot_to_ax(axs[1, 1], pushdown_q95_errors, 'Pushdown q95 errors', max=r1_max)
    plot_to_ax(axs[2, 1], pushdown_q100_errors, 'Pushdown q100 errors', max=r2_max)

    return plt
