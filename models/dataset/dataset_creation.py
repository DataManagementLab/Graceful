import functools
import json
import os.path
import random
from collections import defaultdict
from json import JSONDecodeError
from typing import List, Optional, Any, Dict

import networkx as nx
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from tabulate import tabulate
from torch.utils.data import DataLoader

from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.utils import load_json
from cross_db_benchmark.datasets.datasets import source_dataset_list
from models.dataset.plan_dataset import PlanDataset
from models.dataset.plan_graph_batching.plan_batchers import plan_collator_dict


class NoPlansFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)


def read_workload_runs(workload_run_paths, min_runtime_ms: int, limit_queries=None, limit_queries_affected_wl=None,
                       train_udf_graph_against_udf_runtime: bool = False, stratify_per_database: bool = False,
                       max_runtime: int = None, stratification_prioritize_loops: bool = False):
    # reads several workload runs
    plans = []
    database_statistics = dict()

    min_runtime_discards = 0

    for i, source in enumerate(workload_run_paths):
        try:
            run = load_json(source)
        except JSONDecodeError:
            raise ValueError(f"Error reading {source}")
        database_statistics[i] = run.database_stats
        database_statistics[i].run_kwars = run.run_kwargs

        # extract the database name from the string of the workloads
        database_name = source.split("/")[-1].split(".json")[0]
        # test if file is named after the database - could also be workload.json (and the folder name determines the database)
        if database_name in [ds.name for ds in source_dataset_list]:
            database_statistics[i].database_name = database_name
        else:
            # extract database from folder name
            database_name = source.split("/")[-2]
            database_statistics[i].database_name = database_name

        # get the udf timings
        path, workload_run_file = os.path.split(source)
        path, dataset = os.path.split(path)
        prefix, parsed_plans_folder = os.path.split(path)
        assert parsed_plans_folder in ['parsed_plans', 'deepdb_augmented',
                                       'udf_sel_augmented'], f"Expected parsed_plans_folder to be parsed_plans, but got {parsed_plans_folder}"
        udf_timings_path = os.path.join(prefix, 'dbs', dataset, f'workload_100k_s1_udf_udf_timings.json')

        if train_udf_graph_against_udf_runtime:
            assert os.path.exists(udf_timings_path), f"UDF timings file {udf_timings_path} does not exist"
        if os.path.exists(udf_timings_path):
            with open(udf_timings_path, 'r') as json_file:
                udf_timings = json.load(json_file)
        else:
            udf_timings = None

        limit_per_ds = None
        if limit_queries is not None:
            if i >= len(workload_run_paths) - limit_queries_affected_wl:
                limit_per_ds = limit_queries // limit_queries_affected_wl
                print(f"Capping workload {source} after {limit_per_ds} queries")

        ds_plans = []

        for p_id, plan in enumerate(run.parsed_plans):
            plan.database_name = database_statistics[i].database_name
            plan.database_id = i
            plan.source_file = source
            if '_pullup' in source:
                plan.udf_pullup = True
            else:
                plan.udf_pullup = False
            if udf_timings is not None:
                try:
                    udf_runtimes = udf_timings[plan.udf.udf_name]['runtimes']
                    if isinstance(udf_runtimes, list):
                        # average runtimes
                        udf_runtime = np.mean(udf_runtimes)
                    elif udf_runtimes == -1:  # check for timeout
                        if train_udf_graph_against_udf_runtime:
                            continue
                        else:
                            udf_runtime = None
                    else:
                        raise Exception(f"Unknown runtime format {type(udf_runtimes)} / {udf_runtimes}")
                    plan.udf_timing = udf_runtime
                except KeyError as e:
                    if train_udf_graph_against_udf_runtime:
                        print(f"UDF {plan.udf.udf_name} not found in udf_timings {udf_timings_path} \n {udf_timings}")
                        raise e

            if plan.plan_runtime_ms < min_runtime_ms:
                min_runtime_discards += 1
                continue

            ds_plans.append(plan)
            if limit_per_ds is not None and p_id > limit_per_ds:
                print("Stopping now")
                break

        # remove plans where col to col comparison occurs in the udf,
        # because we have no means to estimate their cardinality yet
        if not 'no_udf' in source and not 'noudf' in source:
            dataset = os.path.basename(os.path.dirname(source))
            exp_path = os.path.dirname(os.path.dirname(os.path.dirname(source)))
            col_col_stats_path = os.path.join(exp_path, 'dbs', dataset, 'created_graphs',
                                              'udf_w_col_col_comparison.json')
            with open(col_col_stats_path, 'r') as f:
                fn_blacklist = json.load(f)

            filtered_plans = []
            for plan in ds_plans:
                if plan.udf.udf_name not in fn_blacklist:
                    filtered_plans.append(plan)

            ds_plans = filtered_plans

        # stratify database by runtimes - do not apply this for additional datasets which are underrepresented
        if stratify_per_database and not '_pullup' in source and not (
                '_no_udf' in source and not '_large' in source) and not '_monitoring' in source:
            assert max_runtime is not None
            ds_plans = balance_plans(ds_plans, max_runtime, hint=source, print_distr_stats=False,
                                     train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
                                     prioritize_loops=stratification_prioritize_loops)
        plans.extend(ds_plans)

    print(f"No of Plans: {len(plans)} (min runtime discards: {min_runtime_discards})")

    return plans, database_statistics


def _inv_log1p(x):
    return np.exp(x) - 1


def balance_plans(plans: List, max_runtime: int, hint: str, print_distr_stats: bool = False,
                  shuffle_before: bool = True, train_udf_graph_against_udf_runtime: bool = False,
                  prioritize_loops: bool = True):
    if len(plans) == 0:
        return plans

    # try to get a uniform distribution of runtimes

    orig_num_plans = len(plans)
    assert max_runtime is not None

    # shuffle the plans with seed 42
    if shuffle_before:
        random.Random(0).shuffle(plans)

    # get the runtimes
    if train_udf_graph_against_udf_runtime:
        runtimes_s = np.array([p.udf_timing for p in plans])
    else:
        runtimes_s = np.array([p.plan_runtime_ms / 1000 for p in plans])
    # get the number of bins
    no_bins = 30
    # get the bin edges
    bin_edges = np.linspace(0, max_runtime - max_runtime / no_bins, no_bins)

    # get the bin ids for each runtime
    bin_ids = np.digitize(runtimes_s, bin_edges, right=False) - 1
    # get the number of plans in each bin
    no_plans_per_bin = np.bincount(bin_ids, minlength=no_bins)
    # get the number of plans in the smallest bin
    min_no_plans_per_bin = np.min(no_plans_per_bin)
    max_no_plans_per_bin = np.max(no_plans_per_bin)

    # move the plans to the bins
    plans_per_bin = [[] for _ in range(no_bins)]
    for plan, bin_id in zip(plans, bin_ids):
        plans_per_bin[bin_id].append(plan)

    udf_sources_dict = dict()

    # keep min_no_plans and at least 10% of the plans in each bin
    no_plans_per_bin_balanced = []
    loop_perc = []
    branch_perc = []
    overall_loops = []
    overall_branches = []
    for i in range(no_bins):
        # keep at least 10 plans
        # keep at least 10% of the plans of the largest bin
        # keep at least min_no_plans_per_bin
        num_plans_to_keep = min(max([min_no_plans_per_bin, int(0.1 * max_no_plans_per_bin), 10]),
                                len(plans_per_bin[i]))

        plans = plans_per_bin[i]

        # extract num loops and num branches
        num_loops = []
        num_branches = []
        for p in plans:
            if hasattr(p, 'udf_num_loops'):
                num_loops.append(p.udf_num_loops)
                num_branches.append(p.udf_num_branches)
            elif not hasattr(p, 'udf'):
                num_loops.append(0)
                num_branches.append(0)
            else:
                source_file = p.source_file
                udf_path = os.path.dirname(os.path.dirname(os.path.dirname(source_file)))
                full_path = os.path.join(udf_path, 'dbs', p.database_name, 'sql_scripts', 'udfs.json')

                if full_path not in udf_sources_dict:
                    with open(full_path, 'r') as f:
                        udf_sources_dict[full_path] = json.load(f)

                # load the udf code
                udf_code_dict = udf_sources_dict[full_path]

                code = udf_code_dict[p.udf.udf_name]
                tmp_l = 0
                tmp_b = 0
                for line in code:
                    if line.strip().startswith('for'):
                        tmp_l += 1
                    if line.strip().startswith('while'):
                        tmp_l += 1
                    if line.strip().startswith('if'):
                        tmp_b += 1
                num_loops.append(tmp_l)
                num_branches.append(tmp_b)

        assert len(num_loops) == len(num_branches) == len(
            plans), f"Lengths do not match: {len(num_loops)}, {len(num_branches)}, {len(plans)}"

        if prioritize_loops:
            # sort by num loops and num branches - keep the plans with the most loops and branches to boost the diversity
            plans = [p for _, _, p in
                     sorted(zip(num_loops, num_branches, plans), key=lambda pair: (pair[0], pair[1]), reverse=True)]

        plans_per_bin[i] = plans[:num_plans_to_keep]
        no_plans_per_bin_balanced.append(num_plans_to_keep)

        loop_perc.append(np.mean(num_loops[:num_plans_to_keep]))
        branch_perc.append(np.mean(num_branches[:num_plans_to_keep]))

        overall_loops.extend(num_loops[:num_plans_to_keep])
        overall_branches.extend(num_branches[:num_plans_to_keep])

    # get the plans
    plans = []
    for i in range(no_bins):
        plans.extend(plans_per_bin[i])

    assert len(bin_edges) == len(no_plans_per_bin) == len(
        no_plans_per_bin_balanced), f"Lengths do not match: {len(bin_edges)}, {len(no_plans_per_bin)}, {len(no_plans_per_bin_balanced)}"

    random.Random(0).shuffle(plans)

    print(f'Balancing of dataset by runtimes: {orig_num_plans} -> {len(plans)} ({hint})', flush=True)
    if print_distr_stats:
        tabulate_data = zip(bin_edges, no_plans_per_bin, no_plans_per_bin_balanced, loop_perc, branch_perc)
        head = ['Bin edges', 'No plans per bin', 'No plans per bin (balanced)', 'Avg. Num Loops', 'Avg. Num Branches']
        print(tabulate(tabulate_data, headers=head))
        print(f"Overall loops: {np.mean(overall_loops):.3f}, overall branches: {np.mean(overall_branches):.3f}")

    return plans


def create_datasets(workload_run_paths, cap_training_samples=None, val_ratio=0.15, limit_queries=None,
                    limit_queries_affected_wl=None, shuffle_before_split=True, loss_class_name=None,
                    stratify_dataset_by_runtimes: bool = False,
                    stratify_per_database_by_runtimes: bool = False,
                    max_runtime: int = None, min_runtime_ms: int = 100,
                    zs_paper_dataset: bool = False, train_udf_graph_against_udf_runtime: bool = False,
                    infuse_plans=None, infuse_database_statistics=None, stratification_prioritize_loops: bool = False,
                    filter_plans: Dict[str, int] = None, w_loop_end_node: bool = True,
                    card_est_assume_lazy_eval: bool = False, ):
    if infuse_plans is not None:
        plans = infuse_plans
        database_statistics = infuse_database_statistics
    else:
        plans, database_statistics = read_workload_runs(workload_run_paths, limit_queries=limit_queries,
                                                        limit_queries_affected_wl=limit_queries_affected_wl,
                                                        train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
                                                        min_runtime_ms=min_runtime_ms,
                                                        stratify_per_database=stratify_per_database_by_runtimes,
                                                        max_runtime=max_runtime,
                                                        stratification_prioritize_loops=stratification_prioritize_loops)

    # extract filter conditions
    if filter_plans is None:
        filter_plans = defaultdict(lambda: None)
    min_num_branches = filter_plans['min_num_branches']
    min_num_loops = filter_plans['min_num_loops']
    max_num_branches = filter_plans['max_num_branches']
    max_num_loops = filter_plans['max_num_loops']
    min_num_np_calls = filter_plans['min_num_np_calls']
    max_num_np_calls = filter_plans['max_num_np_calls']
    min_num_math_calls = filter_plans['min_num_math_calls']
    max_num_math_calls = filter_plans['max_num_math_calls']
    min_num_comp_nodes = filter_plans['min_num_comp_nodes']
    max_num_comp_nodes = filter_plans['max_num_comp_nodes']

    udf_sources_dict = dict()
    filtered_plans = []
    for plan in plans:
        if hasattr(plan, 'udf'):
            if hasattr(plan.udf, 'udf_num_branches'):
                num_branches = plan.udf.udf_num_branches
                num_loops = plan.udf.udf_num_loops
            else:
                source_file = plan.source_file
                udf_path = os.path.dirname(os.path.dirname(os.path.dirname(source_file)))
                full_path = os.path.join(udf_path, 'dbs', plan.database_name, 'sql_scripts', 'udfs.json')

                if full_path not in udf_sources_dict:
                    with open(full_path, 'r') as f:
                        udf_sources_dict[full_path] = json.load(f)

                # load the udf code
                udf_code_dict = udf_sources_dict[full_path]

                code = udf_code_dict[plan.udf.udf_name]
                num_loops = 0
                num_branches = 0
                for line in code:
                    if line.strip().startswith('for'):
                        num_loops += 1
                    if line.strip().startswith('while'):
                        num_loops += 1
                    if line.strip().startswith('if'):
                        num_branches += 1

            # filter plans based on the number of branches and loops
            skip = False
            if min_num_branches is not None and num_branches < min_num_branches:
                skip = True
            elif max_num_branches is not None and num_branches > max_num_branches:
                skip = True
            if min_num_loops is not None and num_loops < min_num_loops:
                skip = True
            elif max_num_loops is not None and num_loops > max_num_loops:
                skip = True

            # filter plans based on the number of np calls
            if min_num_np_calls is not None and plan.udf.udf_num_np_calls < min_num_np_calls:
                skip = True
            elif max_num_np_calls is not None and plan.udf.udf_num_np_calls > max_num_np_calls:
                skip = True

            # filter plans based on the number of math calls
            if min_num_math_calls is not None and plan.udf.udf_num_math_calls < min_num_math_calls:
                skip = True
            elif max_num_math_calls is not None and plan.udf.udf_num_math_calls > max_num_math_calls:
                skip = True

            if not skip and (min_num_comp_nodes is not None or max_num_comp_nodes is not None):
                # filter plans based on the number of computational nodes
                # load the udf graph

                udf_suffix = []
                if w_loop_end_node:
                    udf_suffix.append('loopend')
                if card_est_assume_lazy_eval:
                    udf_suffix.append('lazy')

                if len(udf_suffix) > 0:
                    udf_suffix = '_'.join(udf_suffix)
                    udf_suffix = f'.{udf_suffix}'
                else:
                    udf_suffix = ''

                udf_path = os.path.dirname(os.path.dirname(os.path.dirname(plan.source_file)))

                udf_name = plan.udf.udf_name
                full_path = os.path.join(udf_path, 'dbs', plan.database_name, 'created_graphs',
                                         udf_name + f"{udf_suffix}.gpickle")

                graph = nx.read_gpickle(full_path)

                num_comp_nodes = len([n for n in graph.nodes if graph.nodes[n]['type'] == 'COMP'])

                if min_num_comp_nodes is not None and num_comp_nodes < min_num_comp_nodes:
                    skip = True
                elif max_num_comp_nodes is not None and num_comp_nodes > max_num_comp_nodes:
                    skip = True

            if skip:
                continue

        filtered_plans.append(plan)

        plans = filtered_plans

    if len(plans) == 0:
        raise NoPlansFoundException(f"No plans found in the workload runs: {workload_run_paths}")

    # balance the dataset by the plan runtimes
    if stratify_dataset_by_runtimes:
        plans = balance_plans(plans, max_runtime, hint='all', print_distr_stats=False,
                              train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
                              prioritize_loops=stratification_prioritize_loops)

    assert len(plans) > 0, f"No plans found in the workload runs after stratifying: {workload_run_paths}"

    no_plans = len(plans)
    plan_idxs = list(range(no_plans))
    if shuffle_before_split:
        np.random.shuffle(plan_idxs)

    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_idxs = plan_idxs[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if cap_training_samples is not None:
        prev_train_length = len(train_idxs)
        train_idxs = train_idxs[:cap_training_samples]
        replicate_factor = max(prev_train_length // len(train_idxs), 1)
        train_idxs = train_idxs * replicate_factor

    train_dataset = PlanDataset([plans[i] for i in train_idxs], train_idxs)

    train_udf_only_idxs = [i for i in train_idxs if hasattr(plans[i], 'udf')]
    train_dataset_udf_only = PlanDataset([plans[i] for i in train_udf_only_idxs], train_udf_only_idxs)

    val_dataset = None
    val_dataset_udf_only = None
    if val_ratio > 0:
        val_idxs = plan_idxs[split_train:]
        val_dataset = PlanDataset([plans[i] for i in val_idxs], val_idxs)
        val_udf_only_idxs = [i for i in val_idxs if hasattr(plans[i], 'udf')]
        val_dataset_udf_only = PlanDataset([plans[i] for i in val_udf_only_idxs], val_udf_only_idxs)

    # derive label normalization
    if train_udf_graph_against_udf_runtime:
        runtimes_s = np.array([p.udf_timing for p in plans])
    else:
        if zs_paper_dataset:
            runtimes_s = np.array([p.plan_runtime / 1000 for p in plans])
        else:
            runtimes_s = np.array([p.plan_runtime_ms / 1000 for p in plans])
    label_norm = derive_label_normalizer(loss_class_name, runtimes_s)

    return label_norm, train_dataset, val_dataset, train_dataset_udf_only, val_dataset_udf_only, database_statistics


def derive_label_normalizer(loss_class_name, y):
    if len(y) == 0:
        return None

    if loss_class_name == 'MSELoss':
        log_transformer = preprocessing.FunctionTransformer(np.log1p, _inv_log1p, validate=True)
        scale_transformer = preprocessing.MinMaxScaler()
        pipeline = Pipeline([("log", log_transformer), ("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))
    elif loss_class_name == 'QLoss':
        scale_transformer = preprocessing.MinMaxScaler(feature_range=(1e-2, 1))
        pipeline = Pipeline([("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))
    else:
        pipeline = None
    return pipeline


def rewrite_card_type(card_type: str):
    assert card_type in ['est', 'act', 'dd', 'wj'], f"Unknown card_type_above_udf {card_type}"
    if card_type == 'est':
        rewritten_card_type = 'est_card'
    elif card_type == 'act':
        rewritten_card_type = 'act_card'
    elif card_type == 'dd':
        rewritten_card_type = 'dd_est_card'
    elif card_type == 'wj':
        rewritten_card_type = 'wj_est_card'
    else:
        raise Exception(f"Unknown card_type {card_type}")

    return rewritten_card_type


def create_dataloader(workload_run_paths, test_workload_run_paths: Optional[List[str]], statistics_file, featurization,
                      database: DatabaseSystem, offset_np_import: int,
                      multi_label_keep_duplicates: bool,
                      card_type_below_udf: str,
                      card_type_above_udf: str,
                      card_type_in_udf: str,
                      val_ratio=0.15, batch_size=32, shuffle=True, num_workers=1, pin_memory=False,
                      limit_queries=None, limit_queries_affected_wl=None, loss_class_name=None,
                      stratify_dataset_by_runtimes: bool = False, stratify_per_database_by_runtimes: bool = False,
                      stratification_prioritize_loops: bool = False,
                      max_runtime: int = None, finetune_ratio: float = 0.0,
                      zs_paper_dataset: bool = False, train_udf_graph_against_udf_runtime: bool = False,
                      w_loop_end_node: bool = False, add_loop_loopend_edge: bool = False,
                      card_est_assume_lazy_eval: bool = True, min_runtime_ms: int = 100,
                      est_card_udf_sel: Optional[int] = None, create_dataset_fn_test_artefacts=None,
                      feature_statistics: dict = None, plans_have_no_udf: bool = False, skip_udf: bool = False,
                      filter_plans: Dict[str, int] = None, separate_sql_udf_graphs: bool = False,
                      annotate_flat_vector_udf_preds: bool = False, flat_vector_model_path: str = None,
                      ) -> [Any, Dict, DataLoader, DataLoader, List[DataLoader], List[str], List[DataLoader]]:
    """
    Creates dataloaders that batches physical plans to train the model in a distributed fashion.
    :param workload_run_paths:
    :param val_ratio:
    :param batch_size:
    :param shuffle:
    :param num_workers:
    :param pin_memory:
    :param est_card_udf_sel: adjust all estimated cardinalities by this selectivity constant for a udf-filter (selectivity in %)
    :return:
    """
    # rewrite card types
    rewritten_card_type_below_udf = rewrite_card_type(card_type_below_udf)
    rewritten_card_type_above_udf = rewrite_card_type(card_type_above_udf)

    # split plans into train/test/validation
    if len(workload_run_paths) > 0:
        label_norm, train_dataset, val_dataset, train_dataset_udf_only, val_dataset_udf_only, database_statistics = create_datasets(
            workload_run_paths,
            loss_class_name=loss_class_name,
            val_ratio=val_ratio,
            limit_queries=limit_queries,
            limit_queries_affected_wl=limit_queries_affected_wl,
            stratify_dataset_by_runtimes=stratify_dataset_by_runtimes,
            stratify_per_database_by_runtimes=stratify_per_database_by_runtimes,
            max_runtime=max_runtime,
            zs_paper_dataset=zs_paper_dataset,
            train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
            min_runtime_ms=min_runtime_ms, stratification_prioritize_loops=stratification_prioritize_loops,
            filter_plans=filter_plans, w_loop_end_node=w_loop_end_node,
            card_est_assume_lazy_eval=card_est_assume_lazy_eval)
    else:
        label_norm = None
        train_dataset = None
        train_dataset_udf_only = None
        val_dataset = None
        val_dataset_udf_only = None
        database_statistics = None

    # postgres_plan_collator does the heavy lifting of creating the graphs and extracting the features and thus requires both
    # database statistics but also feature statistics
    if feature_statistics is None:
        feature_statistics = load_json(statistics_file, namespace=False)

    # add stats for artificial features (additional flags / ...)
    feature_statistics['on_udf'] = {"value_dict": {"True": 0, "False": 1}, "no_vals": 2, "type": "categorical"}

    plan_collator = plan_collator_dict[database]
    print(f'database: {database.value}', flush=True)
    dbms = database.value
    assert dbms is not None
    train_collate_fn = functools.partial(plan_collator, db_statistics=database_statistics,
                                         feature_statistics=feature_statistics,
                                         featurization=featurization,
                                         dbms=dbms, offset_np_import=offset_np_import,
                                         multi_label_keep_duplicates=multi_label_keep_duplicates,
                                         zs_paper_dataset=zs_paper_dataset,
                                         train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
                                         w_loop_end_node=w_loop_end_node, add_loop_loopend_edge=add_loop_loopend_edge,
                                         card_est_assume_lazy_eval=card_est_assume_lazy_eval,
                                         est_card_udf_sel=est_card_udf_sel,
                                         card_type_below_udf=rewritten_card_type_below_udf,
                                         card_type_above_udf=rewritten_card_type_above_udf,
                                         card_type_in_udf=card_type_in_udf, plans_have_no_udf=plans_have_no_udf,
                                         skip_udf=skip_udf, separate_sql_udf_graphs=separate_sql_udf_graphs,
                                         annotate_flat_vector_udf_preds=annotate_flat_vector_udf_preds, flat_vector_model_path=flat_vector_model_path)
    dataloader_args = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=train_collate_fn,
                           pin_memory=pin_memory)
    if train_dataset is None or len(train_dataset) == 0:
        train_loader = None
        val_loader = None
        train_loader_udf_only = None
        val_loader_udf_only = None
    else:
        train_loader = DataLoader(train_dataset, **dataloader_args)
        val_loader = DataLoader(val_dataset, **dataloader_args)
        train_loader_udf_only = DataLoader(train_dataset_udf_only, **dataloader_args)
        val_loader_udf_only = DataLoader(val_dataset_udf_only, **dataloader_args)

    # for each test workoad run create a distinct test loader
    test_loaders = None
    test_loader_names = None
    finetune_loaders = None
    if test_workload_run_paths is not None:
        test_loaders = []
        test_loader_names = []
        finetune_loaders = []

        skipped_test_workload_run_paths = []
        for p in test_workload_run_paths:
            if create_dataset_fn_test_artefacts is not None:
                _, test_dataset, finetune_dataset, test_database_statistics = create_dataset_fn_test_artefacts[p]
            else:
                try:
                    _, test_dataset, finetune_dataset, _, _, test_database_statistics = create_datasets([p],
                                                                                                        loss_class_name=loss_class_name,
                                                                                                        val_ratio=finetune_ratio,
                                                                                                        shuffle_before_split=False,
                                                                                                        stratify_dataset_by_runtimes=stratify_dataset_by_runtimes or stratify_per_database_by_runtimes,
                                                                                                        # avoid double stratification. Perform only once since we only have one database anyways here
                                                                                                        max_runtime=max_runtime,
                                                                                                        zs_paper_dataset=zs_paper_dataset,
                                                                                                        train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
                                                                                                        min_runtime_ms=min_runtime_ms,
                                                                                                        stratification_prioritize_loops=stratification_prioritize_loops,
                                                                                                        w_loop_end_node=w_loop_end_node,
                                                                                                        card_est_assume_lazy_eval=card_est_assume_lazy_eval)
                except NoPlansFoundException as e:
                    print(e)
                    print(f"No plans found in test workload run {p}")
                    skipped_test_workload_run_paths.append(p)
                    continue

            # check which cardinalities are available

            entry0 = test_dataset[0][1]
            if not hasattr(entry0.plan_parameters, rewritten_card_type_above_udf) or not hasattr(entry0.plan_parameters,
                                                                                                 rewritten_card_type_below_udf):
                print(f"Skipping test workload run {p} because of missing cardinalities")
                skipped_test_workload_run_paths.append(p)
                continue

            # test dataset
            test_collate_fn = functools.partial(plan_collator, db_statistics=test_database_statistics,
                                                feature_statistics=feature_statistics,
                                                featurization=featurization,
                                                dbms=dbms,
                                                offset_np_import=offset_np_import,
                                                multi_label_keep_duplicates=multi_label_keep_duplicates,
                                                zs_paper_dataset=zs_paper_dataset,
                                                train_udf_graph_against_udf_runtime=train_udf_graph_against_udf_runtime,
                                                w_loop_end_node=w_loop_end_node,
                                                add_loop_loopend_edge=add_loop_loopend_edge,
                                                card_est_assume_lazy_eval=card_est_assume_lazy_eval,
                                                est_card_udf_sel=est_card_udf_sel,
                                                card_type_above_udf=rewritten_card_type_above_udf,
                                                card_type_in_udf=card_type_in_udf,
                                                card_type_below_udf=rewritten_card_type_below_udf,
                                                plans_have_no_udf=plans_have_no_udf, skip_udf=skip_udf,
                                                separate_sql_udf_graphs=separate_sql_udf_graphs,
                                                flat_vector_model_path=flat_vector_model_path,
                                                annotate_flat_vector_udf_preds=annotate_flat_vector_udf_preds)

            # previously shuffle=False but this resulted in bugs
            dataloader_args.update(collate_fn=test_collate_fn)
            test_loader = DataLoader(test_dataset, **dataloader_args)
            test_loaders.append(test_loader)
            test_loader_names.append(p)

            if finetune_ratio > 0:
                finetune_loader = DataLoader(finetune_dataset, **dataloader_args)
                finetune_loaders.append(finetune_loader)

        if len(skipped_test_workload_run_paths) > 0:
            for p in skipped_test_workload_run_paths:
                test_workload_run_paths.remove(p)

    return label_norm, feature_statistics, train_loader, val_loader, train_loader_udf_only, val_loader_udf_only, test_loaders, test_loader_names, finetune_loaders
