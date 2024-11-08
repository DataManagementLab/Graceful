import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import xgboost
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cross_db_benchmark.datasets.datasets import dataset_list_dict
from models.zero_shot_models.utils.losses import QLoss


def extract_feats(udf_graph):
    num_loops = 0
    num_branches = 0
    ops_ctr = defaultdict(int)

    for n in udf_graph.nodes:
        node_data = udf_graph.nodes[n]
        if node_data['type'] in ['VAR', 'INVOCATION', 'RETURN', 'LOOP_END']:
            continue
        elif node_data['type'] == 'LOOP_HEAD':
            num_loops += 1
        elif node_data['type'] == 'BRANCH':
            num_branches += 1
        elif node_data['type'] == 'COMP':
            for op in node_data['ops']:
                ops_ctr[op] += 1

            if node_data['lib_onehot'] != 'null':
                ops_ctr[node_data['lib_onehot']] += 1
        else:
            raise ValueError(f'Unknown type {node_data["type"]} - {node_data}')

    # print(f'Num loops: {num_loops}, num branches: {num_branches}, ops ctr: {ops_ctr}')
    return num_loops, num_branches, ops_ctr


def compute_qerror_perc(pred, true):
    errors = np.maximum(pred / true, true / pred)

    # compute median, 95th, 99th and max error
    errors.sort()
    median = errors[len(errors) // 2]
    avg = sum(errors) / len(errors)
    p95 = errors[int(len(errors) * 0.95)]
    p99 = errors[int(len(errors) * 0.99)]
    max_error = errors[-1]

    return dict(
        avg=avg,
        median=median,
        p95=p95,
        p99=p99,
        max_error=max_error
    )


class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        return data, target


def extract_card_runtime_from_plan(plan):
    assert 'udf_table' in plan['plan_parameters']
    card = plan['plan_parameters']['dd_est_card']

    runtime = plan['plan_runtime_ms']
    return card, runtime


def extract_card_runtime_dict_from_plans(plans, func_names: List[str]):
    udf_card_runtime_dict = defaultdict(list)
    udf_num_loops_branches_dict = dict()

    udf_not_found_set = set()
    for plan in plans['parsed_plans']:
        udf_name = plan['udf']['udf_name']
        if func_names is not None and udf_name not in func_names:
            udf_not_found_set.add(udf_name)
            continue

        udf_card_runtime_dict[udf_name].append(extract_card_runtime_from_plan(plan))

        v = (plan['udf']['udf_num_loops'], plan['udf']['udf_num_branches'])
        if udf_name in udf_num_loops_branches_dict:
            assert udf_num_loops_branches_dict[
                       udf_name] == v, f'Found different values for {udf_name}: {v} vs {udf_num_loops_branches_dict[udf_name]}'
        else:
            udf_num_loops_branches_dict[udf_name] = v

    if func_names is not None and len(udf_not_found_set) > 0:
        print(
            f'{len(udf_not_found_set)}/{len(udf_not_found_set) + len(func_names)} UDFs ignored because no preditions available')

    for udf_name, plans in udf_card_runtime_dict.items():
        # random shuffle
        random.shuffle(plans)
        udf_card_runtime_dict[udf_name] = plans

    return udf_card_runtime_dict, udf_num_loops_branches_dict


api = wandb.Api()

datasets = ['airline', 'imdb', 'ssb', 'tpc_h', 'walmart', 'financial', 'basketball', 'accidents', 'movielens',
            'baseball', 'hepatitis', 'tournament', 'genome', 'credit', 'employee', 'carcinogenesis', 'consumer',
            'geneea', 'seznam', 'fhnk']


def get_table_df(url: str, label: str):
    artifact = api.artifact(url)
    artifact.download()
    table = artifact.get(label)
    return table.get_dataframe()


def extract_dataset_from_name(name: str):
    candidates = []
    for dataset in datasets:
        if name.startswith(f'{dataset}_'):
            candidates.append(dataset)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates for {name}'
    return candidates[0]


if os.path.exists('wandb_inf_df_cache.pkl'):
    import pickle

    with open('wandb_inf_df_cache.pkl', 'rb') as f:
        df_cache = pickle.load(f)
else:
    df_cache = dict()


def get_dataframes_from_run(run: str):
    if run in df_cache:
        print(f'Found run {run} in cache')
        return df_cache[run]

    df_dict = {}

    instance = api.run(f'jwehrstein/udf_cost_est/{run}')
    dataset = extract_dataset_from_name(instance.displayName)

    artifact_list = ['labels']
    for key in instance.summary.keys():
        if key.startswith('preds_'):
            artifact_list.append(key)
    print(artifact_list)
    for key in tqdm(artifact_list):
        df_dict[key] = get_table_df(f'jwehrstein/udf_cost_est/run-{run}-{key}:latest', key)

    df_cache[run] = dataset, df_dict

    # save cache to file
    import pickle
    with open('wandb_inf_df_cache.pkl', 'wb') as f:
        pickle.dump(df_cache, f)

    return dataset, df_dict


def encode_features(feature_list, onehot_all_ops: bool = False, onehot_np_ops: bool = True):
    encoded_feature_list = []

    if onehot_all_ops:
        categories = ['math.erf', 'math.comb', 'math.perm', 'numpy.log', 'math.pow', 'math.cos', 'math.log2', 'encode',
                      'math.log1p', 'isalnum', 'math.log10', 'math.erfc', 'numpy.cos', 'math.fabs', 'isascii', 'find',
                      'numpy.mod', 'upper', 'math.modf', 'numpy.divide', 'lower', 'Div', 'isdecimal', 'endswith',
                      'math.trunc', 'isidentifier', 'math.isqrt', 'isupper', 'isspace', 'numpy.reciprocal', 'center',
                      'istitle', 'replace', 'expandtabs', 'math.tan', 'swapcase', 'title', 'numpy.power',
                      'math.remainder', 'islower', 'math.floor', 'zfill', 'math.exp', 'isalpha', 'math.degrees', 'Sub',
                      'math.ldexp', 'math.lgamma', 'strip', 'math.isfinite', 'math.lcm', 'casefold', 'isnumeric',
                      'numpy.multiply', 'Add', 'numpy.exp', 'math.frexp', 'math.copysign', 'math.gcd', 'math.factorial',
                      'Mult', 'isprintable', 'math.sqrt', 'math.sin', 'isdigit', 'math.isnan', 'math.gamma',
                      'numpy.subtract', 'numpy.sqrt', 'numpy.add', 'math.fmod', 'capitalize', 'math.expm1', 'math.log',
                      'math.ceil', 'math.radians', 'numpy.remainder', 'numpy.sin', 'rfind']
    elif onehot_np_ops:
        categories = ['numpy.log', 'numpy.cos', 'numpy.mod', 'numpy.divide', 'numpy.reciprocal', 'numpy.multiply',
                      'numpy.exp', 'numpy.subtract', 'numpy.sqrt', 'numpy.add', 'numpy.remainder', 'numpy.sin']

    for num_loops, num_branches, ops_ctr in feature_list:
        feats = [num_loops, num_branches]

        if onehot_all_ops:
            for category in categories:
                feats.append(ops_ctr[category])
        else:
            if onehot_np_ops:
                for category in categories:
                    feats.append(ops_ctr[category])

            math_ctr = 0
            np_ctr = 0
            others_ctr = 0
            for key, ctr in ops_ctr.items():
                if key.startswith('math'):
                    math_ctr += ctr
                elif key.startswith('numpy'):
                    np_ctr += ctr
                else:
                    others_ctr += ctr
            feats.extend([math_ctr, np_ctr, others_ctr])

        encoded_feature_list.append(np.asarray(feats))

    return encoded_feature_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, type=str)
    parser.add_argument('--test_against', required=True, type=str)
    parser.add_argument('--model_type', required=True, type=str)

    wb_runs = [
        ('oa65552l', 'tpc_h'),
        ('tbgwbjjj', 'accidents'),
        ('e823aqo7', 'consumer'),
        ('n3hj0bt5', 'geneea'),
        ('a8ul7wfw', 'fhnk'),
        ('a1fkg001', 'imdb'),
        ('q2vbl05x', 'genome'),
        ('nfnfs4zf', 'ssb'),
        ('ocam4ac3', 'financial'),
        # ('myxhaen9', 'airline'),
        # ('06tszwu1', 'carcinogenesis'),
        ('r0si3ki8', 'seznam'),
        ('wyvn07jj', 'credit'),
        ('6wc9ze06', 'baseball'),
        ('3fi7na52', 'basketball'),
        ('rvmwmxnp', 'employee'),
        ('6pbfm8fk', 'movielens'),
        ('wa7bpw7g', 'hepatitis'),
        ('0z3mrqeg', 'walmart'),
        ('c7j8y9yg', 'tournament'),
    ]

    args = parser.parse_args()
    exp_dir = args.exp_dir
    test_against = args.test_against
    model_type = args.model_type

    print(f'Test against {test_against}', flush=True)

    df_dict = None
    for run_id, dataset in wb_runs:
        if dataset == test_against:
            dataset, df_dict = get_dataframes_from_run(run_id)

    assert df_dict is not None

    # extract func name from sql
    tmp = [re.findall(r'func_\d+', sql) for sql in df_dict['labels']['sql']]

    func_names = []
    for entry in tmp:
        assert len(entry) == 1, f'Found {len(entry)} matches in {entry}'
        func_names.append(entry[0])

    act_preds = df_dict['preds_act']['None'].tolist()
    dd_preds = df_dict['preds_dd']['None'].tolist()

    udf_labels_preds_dict = dict()
    for udf_name, label, act_pred, dd_pred in zip(func_names, df_dict['labels']['labels'], act_preds, dd_preds):
        udf_labels_preds_dict[udf_name] = {
            'label': label,
            'act_pred': act_pred,
            'dd_pred': dd_pred
        }

    feature_list = []
    labels_list = []

    test_feature_list = []
    test_labels_list = []
    test_dd_preds_list = []
    test_act_preds_list = []

    for dataset in tqdm(dataset_list_dict['zs_less_scaled']):
        base_dir = os.path.join(exp_dir, 'parsed_plans', dataset.db_name)

        with open(os.path.join(base_dir, 'workload.json'), 'r') as f:
            exp_plans = json.load(f)

        exp_dict, exp_loop_branches_info = extract_card_runtime_dict_from_plans(exp_plans,
                                                                                func_names=func_names if dataset.db_name == test_against else None)

        udf_list = list(exp_dict.keys())

        for udf_name in udf_list:
            full_path = os.path.join(exp_dir, 'dbs', dataset.db_name, 'created_graphs',
                                     udf_name + f".loopend.gpickle")
            assert os.path.exists(full_path), full_path
            udf_graph: nx.DiGraph = nx.read_gpickle(full_path)  # restore UDF graph from gpickle

            if dataset.db_name == test_against:
                test_feature_list.append(extract_feats(udf_graph))
                test_labels_list.append(
                    (udf_labels_preds_dict[udf_name]['label'] * 1000) / exp_dict[udf_name][0][
                        0])  # divide by cardinality
                test_dd_preds_list.append(
                    (udf_labels_preds_dict[udf_name]['dd_pred'] * 1000) / exp_dict[udf_name][0][
                        0])  # divide by cardinality
                test_act_preds_list.append(
                    (udf_labels_preds_dict[udf_name]['act_pred'] * 1000) / exp_dict[udf_name][0][
                        0])  # divide by cardinality
            else:
                feature_list.append(extract_feats(udf_graph))
                labels_list.append(
                    (exp_dict[udf_name][0][1]) / exp_dict[udf_name][0][0])  # divide by cardinality

    onehot_all_ops = False
    onehot_np_ops = True

    encoded_feature_list = encode_features(feature_list, onehot_all_ops=onehot_all_ops, onehot_np_ops=onehot_np_ops)
    encoded_test_feature_list = encode_features(test_feature_list, onehot_all_ops=onehot_all_ops,
                                                onehot_np_ops=onehot_np_ops)

    # shuffle labels and features
    data = list(zip(encoded_feature_list, labels_list))
    random.Random(13).shuffle(data)

    # split in train / test set

    train_data = data[:int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)):]

    train_features, train_labels = zip(*train_data)
    val_features, val_labels = zip(*val_data)

    print(f'Assemble torch tensors', flush=True)

    train_features = torch.tensor(np.asarray(train_features), dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_features = torch.tensor(np.asarray(val_features), dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    test_features = torch.tensor(np.asarray(encoded_test_feature_list), dtype=torch.float32)
    test_labels = torch.tensor(test_labels_list, dtype=torch.float32)

    print(f'Train features: {train_features.shape}, Train labels: {train_labels.shape}', flush=True)
    print(f'Test features: {test_features.shape}, Test labels: {test_labels.shape}', flush=True)

    assert train_features.shape[0] > 0
    assert test_features.shape[0] > 0

    batch_size = 8

    # Create dataset and dataloader
    train_dataset = MyDataset(train_features, train_labels)
    val_dataset = MyDataset(val_features, val_labels)
    test_dataset = MyDataset(test_features, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, output_size)
            self.leaky_relu = nn.LeakyReLU()

        def forward(self, x):
            x = self.leaky_relu(self.fc1(x))
            x = self.leaky_relu(self.fc2(x))
            x = self.leaky_relu(self.fc3(x))
            x = self.fc4(x)
            x = torch.exp(x)
            return x


    print(f'Train on {len(encoded_feature_list)} plans a {model_type} model', flush=True)
    if model_type == 'mlp':

        # Define model, loss function and optimizer
        model = MLP(len(encoded_feature_list[0]), 64, 1)
        print(model, flush=True)
        criterion = QLoss(model=None)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 100

        best_seen_val_loss = float('inf')
        model_state_dict = None

        # Training loop
        for epoch in range(num_epochs):
            train_loss = 0.0
            for inputs, labels in train_loader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}',
                      flush=True)
            if val_loss < best_seen_val_loss:
                best_seen_val_loss = val_loss
                model_state_dict = model.state_dict()

                # print(f'New best seen (epoch: {epoch+1}), Val Loss: {val_loss:.4f}')

            # if epoch == int(num_epochs * 0.75):
            #     # reduce learning rate
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = param_group['lr'] * 0.25

        # load best model
        model.load_state_dict(model_state_dict)

        # Testing the model on the training data
        labels = []
        preds = []
        with torch.no_grad():
            for inputs, lab in test_loader:
                outputs = model(inputs)
                labels.extend(lab)
                preds.extend(outputs.tolist())
    elif model_type == 'xgboost':
        model = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=5, learning_rate=0.1)
        model.fit(encoded_feature_list, labels_list)
        labels = test_labels_list
        preds = model.predict(encoded_test_feature_list)
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # convert to numpy arrays
    labels = np.array(labels)
    preds = np.array(preds)
    preds = preds.reshape(-1)

    # compute qerror

    flat_error = compute_qerror_perc(preds, labels)
    dd_error = compute_qerror_perc(np.asarray(test_dd_preds_list), labels)
    act_error = compute_qerror_perc(np.asarray(test_act_preds_list), labels)

    # plot table
    table_data = []
    for key in flat_error.keys():
        table_data.append([key, flat_error[key], dd_error[key], act_error[key]])

    print(tabulate(table_data, headers=['Metric', 'Flat', 'DD', 'Act']))
