import functools
import os
from collections import defaultdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tabulate

from pull_push_advisor.strategies import greedy_avg, avoid_worst, random_sel, area_under_curve, single_sel, optimal, \
    always_push, conservative, greedy_min, calc_f1, calc_regression


def apply_optimizers(pullup_sql_dict, pullup_dicts, pushdown_dicts, pushdown_labels_dict, pullup_labels_dict):
    sel_list = ['10', '30', '50', '70', '90', 'None']
    # solvers
    solvers = [functools.partial(greedy_avg, sel_list=sel_list),
               functools.partial(avoid_worst, sel_list=sel_list), random_sel,
               functools.partial(area_under_curve, sel_list=sel_list)]
    solver_names = ['greedy_avg', 'avoid_worst', 'auc']

    result_dicts = dict()
    gold_standards_dict = dict()
    selections_dicts = defaultdict(dict)
    baseline_runtimes = dict()

    key_set = set()

    for dataset_key in pullup_sql_dict.keys():
        preds_dict_pullup = pullup_dicts[dataset_key]
        preds_dict_pushdown = pushdown_dicts[dataset_key]
        pullup_labels = pullup_labels_dict[dataset_key]
        pushdown_labels = pushdown_labels_dict[dataset_key]

        pushdown_rt = sum(pushdown_labels)
        baseline_runtimes[dataset_key] = pushdown_rt
        gold_standards_dict[dataset_key] = [a < b for a, b in zip(pullup_labels, pushdown_labels)]

        greedy_fn = functools.partial(greedy_avg, sel_list=sel_list)
        greedy_fn.__name__ = 'greedy_avg'

        auc_fn = functools.partial(area_under_curve, sel_list=sel_list)
        auc_fn.__name__ = 'auc'

        sel_30_fn = functools.partial(single_sel, sel='30')
        sel_30_fn.__name__ = 'single_sel(30)'

        con_fn = functools.partial(conservative, sel_list=sel_list)
        con_fn.__name__ = 'conservative'

        greedymin_fn = functools.partial(greedy_min, sel_list=sel_list)
        greedymin_fn.__name__ = 'greedymin'

        experiments = [
            (single_sel, 'act', 'act'),
            (single_sel, 'wj', 'dd'),
            (single_sel, 'wj', 'wj'),
            (single_sel, 'dd', 'dd'),
            (single_sel, 'est', 'est'),
            (sel_30_fn, 'dd', 'dd'),
            # (sel_30_fn,'wj','dd'),
            (sel_30_fn, 'est', 'est'),
            (optimal, None, None),
            (always_push, None, None),
            (random_sel, None, None),
            (greedy_fn, 'est', 'est'),
            (greedy_fn, 'dd', 'dd'),
            (auc_fn, 'est', 'est'),
            (auc_fn, 'dd', 'dd'),
            # (auc_fn,'wj','dd'),
            (auc_fn, 'wj', 'wj'),
            (con_fn, 'est', 'est'),
            (con_fn, 'dd', 'dd'),
            # (con_fn,'wj','dd'),
            (con_fn, 'wj', 'wj'),
            (greedymin_fn, 'est', 'est'),
            (greedymin_fn, 'dd', 'dd'),
            (greedymin_fn, 'wj', 'wj'),
        ]

        results = dict()
        for entry in experiments:
            if (entry[1] is not None and entry[1] not in preds_dict_pullup) or (
                    entry[2] is not None and entry[2] not in preds_dict_pushdown):
                print(
                    f'Skipping {entry[0].__name__} / {entry[1]} / {entry[2]} for {dataset_key} because of missing data')
                continue

            if entry[1] is None:
                preds_pullup = None
            else:
                assert entry[
                           1] in preds_dict_pullup, f'Key {entry[1]} not in preds_dict_pullup: {preds_dict_pullup.keys()}'
                preds_pullup = preds_dict_pullup[entry[1]]
            if entry[2] is None:
                preds_pushdown = None
            else:
                preds_pushdown = preds_dict_pushdown[entry[2]]

            name = entry[0].__name__ if entry[0].__name__ != '<lambda>' else entry[0].func.__name__
            key = (name, entry[1], entry[2])
            selections, num_pullups, runtime = entry[0](preds_pullup, preds_pushdown, pullup_labels, pushdown_labels)

            precision, recall, f1, accuracy, tp_percentage = calc_f1(selections, gold_standards_dict[dataset_key])

            fp_percentage, regression_in_total_runtime_rel, regression_in_total_runtime_abs, if_regression_how_worse_avg = calc_regression(
                selections, gold_standards_dict[dataset_key], pullup_labels_dict[dataset_key],
                pushdown_labels_dict[dataset_key], runtime)

            results[key] = {
                'runtime': runtime,
                'num_pullups': num_pullups,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'impr_over_duckdb': (pushdown_rt - runtime) / pushdown_rt,
                'tp_percentage': tp_percentage,
                'fp_percentage': fp_percentage,
                'regression_in_total_runtime_rel': regression_in_total_runtime_rel,
                'regression_in_total_runtime_abs': regression_in_total_runtime_abs,
                'if_regression_how_worse_avg': if_regression_how_worse_avg
            }

            key_set.add(key)

            assert key not in selections_dicts[dataset_key]
            selections_dicts[dataset_key][key] = selections

        result_dicts[dataset_key] = results

    return result_dicts


def run_plan_advisor(df_dict: Dict, dataset: str):
    # initialize dictionaries
    pullup_dicts = {}
    pushdown_dicts = {}
    pushdown_labels_dict = {}
    pullup_labels_dict = {}
    pushdown_sql_dict = {}
    pullup_sql_dict = {}

    for i, r in enumerate(['this`']):
        assert dataset not in pullup_dicts, f'Dataset {dataset} already in pullup_dicts {pullup_dicts.keys()}'

        labels = df_dict['labels']
        preds_dict_pullup = dict()
        preds_dict_pushdown = dict()
        for key, value in df_dict.items():
            if key.startswith('preds_pullup'):
                preds_dict_pullup[key.replace('preds_pullup_', '')] = value
            elif key.startswith('preds_pushdown'):
                preds_dict_pushdown[key.replace('preds_pushdown_', '')] = value

        pullup_dicts[dataset] = preds_dict_pullup
        pushdown_dicts[dataset] = preds_dict_pushdown

        labels = dict(labels)

        pushdown_labels_dict[dataset] = labels['pushdown']
        pullup_labels_dict[dataset] = labels['pullup']
        pushdown_sql_dict[dataset] = labels['pushdown_sql']
        pullup_sql_dict[dataset] = labels['pullup_sql']

    result_dicts = apply_optimizers(
        pullup_sql_dict=pullup_sql_dict,
        pullup_dicts=pullup_dicts,
        pushdown_dicts=pushdown_dicts,
        pushdown_labels_dict=pushdown_labels_dict,
        pullup_labels_dict=pullup_labels_dict
    )

    datasets = []

    speedup_dict = defaultdict(dict)
    acc_dict = defaultdict(dict)
    tp_percentage = defaultdict(dict)
    fp_percentage = defaultdict(dict)
    regression_overhead = defaultdict(dict)
    runtime = defaultdict(dict)

    def process(key: Tuple, card: str, solver: str, label: str, results: Dict, dataset: str):
        out_key = (card, solver, label)
        if key not in results:
            speedup_dict[out_key][dataset] = np.nan
            acc_dict[out_key][dataset] = np.nan
            tp_percentage[out_key][dataset] = np.nan
            fp_percentage[out_key][dataset] = np.nan
            regression_overhead[out_key][dataset] = np.nan
            runtime[out_key][dataset] = np.nan
        else:
            speedup_dict[out_key][dataset] = results[key]['impr_over_duckdb'] + 1  # convert improvement to speedup
            acc_dict[out_key][dataset] = results[key]['accuracy']
            tp_percentage[out_key][dataset] = results[key]['tp_percentage']
            fp_percentage[out_key][dataset] = results[key]['fp_percentage']
            regression_overhead[out_key][dataset] = results[key]['regression_in_total_runtime_rel']
            runtime[out_key][dataset] = results[key]['runtime']

    for dataset in sorted(list(pullup_sql_dict.keys())):
        datasets.append(dataset)

        results = result_dicts[dataset]

        # process(('single_sel', 'est', 'est'), 'DuckDB','Cost', results, dataset)
        # process(('conservative', 'est', 'est'), 'DuckDB','Conservative', results=results, dataset=dataset)
        process(('single_sel', 'dd', 'dd'), 'DeepDB', 'UBC', 'GRACEFUL (UBC, DeepDB Card)', results, dataset=dataset)
        process(('auc', 'dd', 'dd'), 'DeepDB', 'AuC', 'GRACEFUL (AuC, DeepDB Card)', results=results, dataset=dataset)
        process(('conservative', 'dd', 'dd'), 'DeepDB', 'Conservative', 'GRACEFUL (Conservative, DeepDB Card)',
                results=results, dataset=dataset)
        # process(('single_sel', 'wj', 'dd'), 'Card: WJ / DeepDB (Cost)', results=results, dataset=dataset)
        # process(('auc', 'wj', 'dd'), 'Card: WJ / DeepDB (AuC)', results=results, dataset=dataset)
        # process(('conservative', 'wj', 'dd'), 'Card: WJ / DeepDB (Conservative)', results=results, dataset=dataset)
        # process(('single_sel', 'wj', 'wj'), 'WJ','Cost', results=results, dataset=dataset)
        # process(('auc', 'wj', 'wj'), 'WJ','AuC', results=results, dataset=dataset)
        # process(('conservative', 'wj', 'wj'), 'WJ','Conservative', results=results, dataset=dataset)
        # process(('greedymin', 'wj', 'wj'), 'Card: WJ (GreedyMin)', results=results, dataset=dataset)
        process(('single_sel', 'act', 'act'), 'Actual', 'Cost', 'GRACEFUL (Cost, Actual Card)', results=results,
                dataset=dataset)
        process(('optimal', None, None), '', 'Optimal', 'Optimum', results=results, dataset=dataset)
        process(('always_push', None, None), '', 'Always Push', 'Always Push', results=results, dataset=dataset)

    dicts = [speedup_dict]
    metrics = ['Speedup']

    # create barplot with both lists
    plt.close()
    fig, axs = plt.subplots(len(dicts), 1, figsize=(12, len(dicts) * 2))
    barWidth = 0.8 / len(speedup_dict)

    for i in range(len(dicts)):
        value_dict = dicts[i].copy()
        metric = metrics[i]

        ax = axs

        optimum_key = ('', 'Optimal', 'Optimum')
        if optimum_key in value_dict:
            optimum_speedup = value_dict.pop(optimum_key)

        for i, key in enumerate(value_dict.keys()):
            label = key[2]
            ax.bar([x + i * barWidth for x in range(len(value_dict[key]))],
                   [value_dict[key][dataset] for dataset in datasets], width=barWidth, label=label)

        # draw optimum line
        for i, dataset in enumerate(datasets):
            ax.hlines(y=optimum_speedup[dataset], xmin=i - .1, xmax=i + 0.7, colors='black', linestyles='dashed',
                      label='Optimum', linewidth=1)

        # draw line at y = 1
        if metric == 'Speedup':
            ax.axhline(y=1, color='black', linestyle='-', label='No Pullup', linewidth=1)

        # ax.set_xlabel(f'Datasets')
        ax.set_ylabel(f'{metric}')

        ax.set_xticks([r + 0.25 for r in range(len(datasets))])
        ax.set_xticklabels(datasets, rotation=45, ha='right')

        # set title
        # ax.set_title(f'{metric} of different pullup/pushdown advisors compared to DuckDB')

        # set y lim to 1
        # ax.set_ylim(1, max([max(l) for l in runtime_list_dict.values()]) + 0.1)
        if metric == 'Accuracy':
            ax.set_ylim(0, 1)

        # place legend below chart
        handles, plot_labels = ax.get_legend_handles_labels()

        print(len(handles))
        print(len(plot_labels))

        # handles = handles[19:]
        # plot_labels = plot_labels[19:]

        # TODO idx error

        order = [1, 0, 5, 4, 3, 2]
        ax.legend([handles[idx] for idx in order], [plot_labels[idx] for idx in order], loc='upper center',
                  bbox_to_anchor=(0.5, -0.6), shadow=False, ncol=3, frameon=False)
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=3, shadow=False, frameon=False, )

    plt.show()
    # create dir
    os.makedirs('results', exist_ok=True)
    fig.savefig('results/pullup_pushdown_speedup.jpg', bbox_inches='tight')

    tabulate_data = []
    for key, vals in speedup_dict.items():
        vs = [v for v in vals.values() if not np.isnan(v)]
        regression_overheads = [v for v in regression_overhead[key].values() if not np.isnan(v)]
        tp_percentages = [v for v in tp_percentage[key].values() if not np.isnan(v)]
        fp_percentages = [v for v in fp_percentage[key].values() if not np.isnan(v)]
        runtimes = [v for v in runtime[key].values() if not np.isnan(v)]
        tabulate_data.append((key[0], key[1], sum(vs) / len(vs), min(vs), max(vs), np.median(vs),
                              f'{sum(tp_percentages) / len(tp_percentages):.2%}',
                              f'{sum(fp_percentages) / len(fp_percentages):.2%}',
                              f'{sum(regression_overheads) / len(regression_overheads):.2%}', sum(runtimes)))

    tabulate_data.sort(key=lambda x: x[2], reverse=True)

    print(f'Speedup statistics (Avg. over {len(datasets)} datasets)')
    print(tabulate.tabulate(tabulate_data,
                            headers=['Card', 'Solver', 'Avg. Speedup', 'Min Speedup', 'Max Speedup', 'Median Speedup',
                                     'TP Percentage',
                                     'FP Percentage', 'FP Impact (relative to total runtime)', 'Total Runtime']))
