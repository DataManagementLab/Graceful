from typing import Dict, List

import numpy as np


def optimal(preds_dict_pullup, preds_dict_pushdown, pullup_labels, pushdown_labels):
    # select the plan for which the runtime prediction is the lowest
    selections = [a < b for a, b in zip(pullup_labels, pushdown_labels)]

    return selections, sum(selections), sum(
        [a if decision else b for decision, a, b in zip(selections, pullup_labels, pushdown_labels)])


def greedy_avg(preds_dict_pullup, preds_dict_pushdown, pullup_labels, pushdown_labels, sel_list):
    # select the plan for which the avg cost prediction (across all selectivities) is the lowest
    avg_preds_pullup = [sum([preds_dict_pullup[sel][i] for sel in sel_list]) / len(sel_list) for i in
                        range(len(pullup_labels))]
    avg_preds_pushdown = [sum([preds_dict_pushdown[sel][i] for sel in sel_list]) / len(sel_list) for i in
                          range(len(pushdown_labels))]

    selections = [a < b for a, b in zip(avg_preds_pullup, avg_preds_pushdown)]

    return selections, sum(selections), sum(
        [a if decision else b for decision, a, b in zip(selections, pullup_labels, pushdown_labels)])


def avoid_worst(preds_dict_pullup, preds_dict_pushdown, pullup_labels, pushdown_labels, sel_list):
    # select the plan for which the avg cost prediction (across all selectivities) is the lowest
    max_preds_pullup = [max([preds_dict_pullup[sel][i] for sel in sel_list]) for i in
                        range(len(pullup_labels))]
    max_preds_pushdown = [sum([preds_dict_pushdown[sel][i] for sel in sel_list]) / len(sel_list) for i in
                          range(len(pushdown_labels))]

    selections = [a < b for a, b in zip(max_preds_pullup, max_preds_pushdown)]

    return selections, sum(selections), sum(
        [a if decision else b for decision, a, b in zip(selections, pullup_labels, pushdown_labels)])


def single_sel(preds_dict_pullup, preds_dict_pushdown, pullup_labels, pushdown_labels, sel: str = 'None',
               improvement_factor: float = 1.0):
    # select the plan for which the sel50 prediction is the lowest
    sel_preds_pullup = [preds_dict_pullup[sel][i] for i in
                        range(len(pullup_labels))]
    sel_preds_pushdown = [preds_dict_pushdown[sel][i] for i in
                          range(len(pushdown_labels))]

    selections = [a * improvement_factor < b for a, b in zip(sel_preds_pullup, sel_preds_pushdown)]

    return selections, sum(selections), sum(
        [a if decision else b for decision, a, b in zip(selections, pullup_labels, pushdown_labels)])


def random_sel(preds_dict_pullup, preds_dict_pushdown, pullup_labels, pushdown_labels):
    import random
    # set random seed
    random.seed(42)
    selections = [random.choice([True, False]) for _ in range(len(pullup_labels))]

    return selections, sum(selections), sum(
        [a if decision else b for decision, a, b in zip(selections, pullup_labels, pushdown_labels)])


def always_push(preds_dict_pullup, preds_dict_pushdown, pullup_labels, pushdown_labels):
    return [False] * len(pullup_labels), 0, sum(pushdown_labels)


def area_under_curve(preds_dict_pullup, preds_dict_pushdown, pullup_labels, pushdown_labels, sel_list):
    # select the plan for which the area under the curve is the lowest
    assert sel_list[-1] == 'None', 'Last element of sel_list must be None (i.e. sel 1.0)'

    def calc_auc(preds_dict: Dict, sel_list: List[str]):
        area = None

        for i in range(len(sel_list)):
            if i == 0:
                start_key = sel_list[i]
                start_bound = float(sel_list[i]) / 100
            else:
                start_key = sel_list[i - 1]
                start_bound = float(sel_list[i - 1]) / 100

            if i == len(sel_list) - 1:
                end_bound = 1
            else:
                end_bound = float(sel_list[i]) / 100

            end_key = sel_list[i]

            iteration = (end_bound - start_bound) * (
                    np.asarray(preds_dict[end_key]) + np.asarray(preds_dict[start_key])) / 2
            if area is None:
                area = iteration
            else:
                area += iteration

        return area

    pull_aucs = calc_auc(preds_dict_pullup, sel_list)
    push_aucs = calc_auc(preds_dict_pushdown, sel_list)

    selections = [a < b for a, b in zip(pull_aucs, push_aucs)]

    return selections, sum(selections), sum(
        [a if decision else b for decision, a, b in zip(selections, pullup_labels, pushdown_labels)])


def conservative(preds_dict_pullup, preds_dict_pushdown, pullup_labels, pushdown_labels, sel_list):
    # select only a pullup plan if it is faster than the pushdown plan for all selectivities
    pullup_faster = [all([preds_dict_pullup[sel][i] < preds_dict_pushdown[sel][i] for sel in sel_list]) for i in
                     range(len(pullup_labels))]
    selections = pullup_faster

    return selections, sum(selections), sum(
        [a if decision else b for decision, a, b in zip(selections, pullup_labels, pushdown_labels)])


def greedy_min(preds_dict_pullup, preds_dict_pushdown, pullup_labels, pushdown_labels, sel_list):
    # select only a pullup plan if it is faster than the pushdown plan for all selectivities
    selections = [
        min([preds_dict_pullup[sel][i] for sel in sel_list]) < min([preds_dict_pushdown[sel][i] for sel in sel_list])
        for i in range(len(pullup_labels))]

    return selections, sum(selections), sum(
        [a if decision else b for decision, a, b in zip(selections, pullup_labels, pushdown_labels)])


def calc_f1(selections, gold_standard_selections):
    assert len(selections) == len(gold_standard_selections)

    # calculate f1 score
    tp = sum([1 for p, l in zip(selections, gold_standard_selections) if p and l])
    tn = sum([1 for p, l in zip(selections, gold_standard_selections) if not p and not l])
    fp = sum([1 for p, l in zip(selections, gold_standard_selections) if p and not l])
    fn = sum([1 for p, l in zip(selections, gold_standard_selections) if not p and l])

    if tp + fp == 0:
        precision = 1
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    accuracy = (tp + tn) / len(selections)

    return precision, recall, f1, accuracy, tp / len(selections)


def calc_regression(selections, gold_standard_selections, pullup_labels, pushdown_labels, overall_runtime):
    assert len(selections) == len(gold_standard_selections)

    # extract all entries where selections is true but gold_standard_selections is false
    false_positives_runtimes = [pullup_label for sel, true_sel, pullup_label in
                                zip(selections, gold_standard_selections, pullup_labels) if sel and not true_sel]

    # calculate average slowdown of false positives (in percentage)
    false_positives_ideal_runtimes = [pushdown_label for sel, true_sel, pushdown_label in
                                      zip(selections, gold_standard_selections, pushdown_labels) if
                                      sel and not true_sel]

    # regression
    fp_difference = [a - b for a, b in zip(false_positives_runtimes, false_positives_ideal_runtimes)]

    if len(false_positives_runtimes) == 0:
        avg_regression_slowdown_perc = 0
    else:
        avg_regression_slowdown_perc = sum(
            [(a - b) / b for a, b in zip(false_positives_runtimes, false_positives_ideal_runtimes)]) / len(
            false_positives_runtimes)

    # fp percentage, difference by fp choice (regression in %), total sum of regression, avg regression in %
    return len(false_positives_runtimes) / len(selections), sum(fp_difference) / overall_runtime, sum(
        fp_difference), avg_regression_slowdown_perc
