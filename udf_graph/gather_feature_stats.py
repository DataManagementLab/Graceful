import functools
import json
import multiprocessing
import os
import time
from multiprocessing.pool import ThreadPool
from typing import Dict, Any, List

import networkx as nx
import numpy as np
from sklearn.preprocessing import RobustScaler

from cross_db_benchmark.datasets.datasets import dataset_list_dict, Dataset

FEATURE_CLASS_DICT = {
    "numerical": ["in_rows_act", "in_rows_est", 'in_rows_deepdb', "no_params", "no_iter"],
    "categorical": ["in_dts", "ops", "loop_part", "cmops", "cmdtypes", "loop_part", "loop_type", "fixed_iter",
                    "out_dts", "lib_onehot"],
    "vector": ["lib_embedding"]
}

FEATURE_DICT = {
    "in_rows_act": [],
    "in_rows_est": [],
    "in_rows_deepdb": [],
    "in_dts": [],
    "no_params": [],
    "lib_embedding": [],
    "lib_onehot": [],
    "ops": [],
    "loop_part": [],
    "cmops": [],
    "cmdtypes": [],
    "loop_type": [],
    "fixed_iter": [],
    "no_iter": [],
    "out_dts": []
}


def gather_feature_stats(graph, dict):
    for node in graph.nodes:
        # iterate over the attributes of each node
        for key in graph.nodes[node].keys():
            if key in dict.keys():
                if type(graph.nodes[node][key]) == list:
                    dict[key] += graph.nodes[node][key]
                else:
                    dict[key] += [graph.nodes[node][key]]
    return dict


def gather_feature_stats_fn(dataset: Dataset, exp_folder: str, suffix: str = '.loopend') -> Dict[str, List[Any]]:
    try:
        # look which UDFs are in the parsed file
        json_path = os.path.join(exp_folder, 'parsed_plans', dataset.db_name, "workload.json")

        assert os.path.exists(json_path), f'File does not exist: {json_path}'
        print(f'Process: {json_path}', flush=True)
        start_t = time.time()

        # load the json file into a dictionary
        with open(json_path, 'r') as f:
            data = json.load(f)

        feature_dict = FEATURE_DICT.copy()

        def process_plan(plan):
            func_name = plan["udf"]['udf_name']
            graph = nx.read_gpickle(
                os.path.join(exp_folder, 'dbs', dataset.db_name, "created_graphs", func_name + f"{suffix}.gpickle"))
            gather_feature_stats(graph, feature_dict)

        # for plan in data["parsed_plans"]:
        with ThreadPool() as pool:
            pool.map(process_plan, data["parsed_plans"])

        print(f'Finished in {time.time() - start_t:.2f}s: {json_path}', flush=True)

        return feature_dict
    except Exception as e:
        print(e, flush=True)
        raise e


def gather_feature_stats_wrapper(exp_folder, dict, dataset_list_name: str):
    fn = functools.partial(gather_feature_stats_fn, exp_folder=exp_folder)
    with multiprocessing.Pool() as pool:
        result_dict = pool.map(fn, dataset_list_dict[dataset_list_name])

    # merge the dictionaries
    for result in result_dict:
        for key in result.keys():
            if result[key] != []:
                dict[key] += result[key]

    # produce the encoders / scalers / feature stats
    feature_stats_2_json(dict, loc=os.path.join(exp_folder, "parsed_plans"))


def feature_stats_2_json(dict, loc, feat_classes_dict=FEATURE_CLASS_DICT):
    # creation of the JSON was adapted from Benjamins codebase
    feature_stats = {}
    for key in dict.keys():
        if dict[key] != []:  # only consider features that were actually observed
            if key in feat_classes_dict["numerical"]:
                scaler = RobustScaler()
                np_values = np.array(dict[key], dtype=np.float32).reshape(-1, 1)
                scaler.fit(np_values)
                feature_stats[key] = {"max": float(np_values.max()), "scale": scaler.scale_.item(),
                                      "center": scaler.center_.item(), "type": "numeric"}
            elif key in feat_classes_dict["categorical"]:
                unique_values = set(dict[key])
                feature_stats[key] = {"value_dict": {v: id for id, v in enumerate(unique_values)},
                                      "no_vals": len(unique_values), "type": "categorical"}
            elif key in feat_classes_dict['vector']:  # type "vector"
                feature_stats[key] = {"vec_len": len(dict[key][0]), "type": "vector"}
            else:
                raise ValueError(f"Feature {key} was not assigned to a feature class")
        else:
            print(f"Feature {key} was not observed in the graph")
    # save as json
    out_path = os.path.join(loc, "udf_stats.json")
    print(f"Saving udf feature stats to {out_path}")
    with open(out_path, 'w') as outfile:
        json.dump(feature_stats, outfile)
