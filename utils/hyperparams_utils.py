import copy
import os
from typing import Dict, Any

from cross_db_benchmark.datasets.datasets import dataset_list_dict
from models.dataset.plan_featurization.postgres_plan_featurizations import featurization_dict

config_keywords = {
    'qloss': 'loss',
    'mse': 'loss',
    'procloss': 'loss',
    'ddactfonudf': 'plan_featurization',
    'ddact': 'plan_featurization',
    'ddestfonudf': 'plan_featurization',
    'ddest': 'plan_featurization',
    'dddeepfonudf': 'plan_featurization',
    'ddwjfonudf': 'plan_featurization',
    'pgzsdd': 'plan_featurization',
    'pgzsact': 'plan_featurization',
    'liboh': 'featurization',  # add one hot featurization of library call
    'libemb': 'featurization',  # add embedding featurization of library call
    'res': 'featurization',  # add residual connection in MLP layers
    'gradnorm': 'model',  # apply gradient normalizatoin
    'mldupl': 'model',
    # keep duplicates in multi label features (i.e. produce one-hot where multi non-zeros are in, and values are not necessarily 1)
    'nolrred': 'model',  # do not apply lr reduction on plateau
    'loopend': 'model',  # add loop end node
    'loopedge': 'model',  # add edge between loop and loopend
    'lazy': 'model',  # assume lazy evaluation for cardinality estimation
    'gradntemb': 'model',  # allow gradients on embeddings of node-type encoder
}


def get_config(hyperparams: Dict[str, Any], wl_base_path: str, assemble_filenames: bool = True):
    """
    Expects:
    - model_config (str):  with model configuration keywords, separated by _
    - data_keyword (str): which dataset to use
    - max_runtime (int): maximum runtime of queries in workload [default: 30]
    Optional:
    - batch_size (int): batch size [default: 128]
    - epochs (int): number of epochs [default: 100]
    - early_stopping_patience (int): early stopping patience [default: 10]
    - test_against (str): which dataset to test against [default: imdb]
    - train_on_test (bool): whether to train on test dataset [default: False]
    - stratify_dataset_by_runtimes (bool): whether to stratify dataset by runtimes [default: False]
    - mp_ignore_udf (bool): whether to ignore UDFs in message passing [default: False]
    - optimizer (str): which optimizer to use [default: AdamW]
    """

    model_config = hyperparams.pop('model_config')

    # define model name
    model_name = model_config

    # general fc out
    fc_out_kwargs = dict(
        activation_class_name='LeakyReLU',
        activation_class_kwargs={},
        norm_class_name='Identity',
        norm_class_kwargs={},
        residual=False,
        dropout=False,
        p_dropout=0.0,
        activation=True,
        inplace=True)
    final_mlp_kwargs = dict(
        width_factor=1.5,
        n_layers=4,
        input_dim=128,  # graph dim
        loss_class_name='QLoss',
    )
    tree_layer_kwargs = dict(
        tree_layer_name='MscnConv',
        width_factor=0.6,  # tree layer width factor
        n_layers=2,  # message passing layers
        hidden_dim=128,  # graph dim
    )
    node_type_kwargs = dict(
        width_factor=1.5,  # node type width factor
        n_layers=4,  # node layers
        one_hot_embeddings=True,
        output_dim=128,  # graph dim
        max_emb_dim=32,
        drop_whole_embeddings=False,
        allow_gradients_on_embeddings=False
    )

    config = dict(
        optimizer_class_name='AdamW',
        optimizer_kwargs=dict(lr=0.001),
        batch_size=256,
        epochs=50,
        ft_epochs_udf_only=0,
        early_stopping_patience=10,
        max_runtime=None,
        apply_gradient_norm=False,
        mp_ignore_udf=False,  # ignore UDFs in message passing
        multi_label_keep_duplicates=False,
        # keep duplicates in multi label features (i.e. produce one-hot where multi non-zeros are in, and values are not necessarily 1)
        apply_lr_reduction_on_plateau=True,
        zs_paper_dataset=False,
        plans_have_no_udf=False,
        pretrained_model_artifact_dir=None,
        pretrained_model_filename=None,
        train_udf_graph_against_udf_runtime=False,
        work_with_udf_repr=False,
        valtest=True,
        w_loop_end_node=False,
        add_loop_loopend_edge=False,
        card_est_assume_lazy_eval=False,
        test_with_count_edges_msg_aggr=False,
        min_runtime_ms=100,
        stratify_per_database_by_runtimes=False,
        stratify_dataset_by_runtimes=False,
        stratification_prioritize_loops=False,
        offset_np_import=0,
        card_type='act',
        skip_udf=False,
        filter_plans=dict(
            min_num_branches=None,
            max_num_branches=None,
            min_num_loops=None,
            max_num_loops=None,
            min_num_np_calls=None,
            max_num_np_calls=None,
            min_num_math_calls=None,
            max_num_math_calls=None,
            min_num_comp_nodes=None,
            max_num_comp_nodes=None,
        )
    )

    """
    Other keywords:
    - graphdim{int}: set graph dimension to {int}
    - dropout{float}: set dropout probability to {float}
    - offsetnp{int}: set offset for np imports {int} - ms
    """

    if 'zs_paper_dataset' in hyperparams:
        config['zs_paper_dataset'] = hyperparams.get('zs_paper_dataset')

    if 'train_on_test' in hyperparams:
        config['valtest'] = False

    data_wl_str = ''

    if 'include_no_udf_data' in hyperparams:
        assert hyperparams['include_no_udf_data']
        include_no_udf_data = hyperparams.pop('include_no_udf_data')
        assert include_no_udf_data
        data_wl_str += 'noudf'
    else:
        include_no_udf_data = False

    if 'include_pullup_data' in hyperparams:
        assert hyperparams['include_pullup_data']
        include_pullup_data = hyperparams.pop('include_pullup_data')
        assert include_pullup_data
        data_wl_str += 'pullup'
    else:
        include_pullup_data = False

    if 'include_pushdown_data' in hyperparams:
        assert hyperparams['include_pushdown_data']
        include_pushdown_data = hyperparams.pop('include_pushdown_data')
        assert include_pushdown_data
        data_wl_str += 'pushdown'
    else:
        include_pushdown_data = False

    if 'include_no_udf_data_large' in hyperparams:
        assert hyperparams['include_no_udf_data_large']
        include_no_udf_data_large = hyperparams.pop('include_no_udf_data_large')
        assert include_no_udf_data_large
        data_wl_str += 'noudflarge'
    else:
        include_no_udf_data_large = False

    if 'include_select_only_w_branch' in hyperparams:
        assert hyperparams['include_select_only_w_branch']
        include_select_only_w_branch = hyperparams.pop('include_select_only_w_branch')
        assert include_select_only_w_branch
        data_wl_str += 'wbranch'
    else:
        include_select_only_w_branch = False

    model_name = f'{data_wl_str}_{model_name}'

    # compile workload runs / test workload runs
    if assemble_filenames:
        assert data_wl_str != '', 'No data workloads selected'
        train_wl_paths, test_wl_paths, statistics_file, data_keyword, model_name_suffix = compile_train_test_filenames(
            hyperparams, wl_base_path=wl_base_path, zs_paper_dataset=config['zs_paper_dataset'],
            include_no_udf_data=include_no_udf_data, include_pullup_data=include_pullup_data,
            include_pushdown_data=include_pushdown_data, include_no_udf_data_large=include_no_udf_data_large,
            include_select_only_w_branch=include_select_only_w_branch)
    else:
        data_keyword = hyperparams.pop('data_keyword')
        train_wl_paths = None
        test_wl_paths = None
        statistics_file = None
        model_name_suffix = 'notused'
    model_name = f'{data_keyword}_{model_name}_{model_name_suffix}'

    if 'batch_size' in hyperparams:
        batch_size = hyperparams.pop('batch_size')
        config['batch_size'] = batch_size
        model_name += f'_bs{batch_size}'
    if 'stratify_dataset_by_runtimes' in hyperparams:
        config['stratify_dataset_by_runtimes'] = hyperparams.pop('stratify_dataset_by_runtimes')
        model_name += '_stratR'
    if 'stratify_per_database_by_runtimes' in hyperparams and hyperparams['stratify_per_database_by_runtimes']:
        config['stratify_per_database_by_runtimes'] = hyperparams.pop('stratify_per_database_by_runtimes')
        model_name += '_stratDBs'
    if 'stratification_prioritize_loops' in hyperparams and hyperparams['stratification_prioritize_loops']:
        config['stratification_prioritize_loops'] = hyperparams.pop('stratification_prioritize_loops')
        model_name += '_stratPL'
    if 'skip_udf' in hyperparams and hyperparams['skip_udf']:
        config['skip_udf'] = hyperparams.pop('skip_udf')
        model_name = 'skipUDF_' + model_name
    if 'epochs' in hyperparams:
        epochs = hyperparams.pop('epochs')
        config['epochs'] = epochs
        model_name += f'_ep{epochs}'
    if 'ft_epochs_udf_only' in hyperparams:
        ft_epochs_udf_only = hyperparams.pop('ft_epochs_udf_only')
        config['ft_epochs_udf_only'] = ft_epochs_udf_only
        model_name += f'_ftudfep{ft_epochs_udf_only}'
    if 'early_stopping_patience' in hyperparams:
        early_stopping_patience = hyperparams.pop('early_stopping_patience')
        config['early_stopping_patience'] = early_stopping_patience
        model_name += f'_esp{early_stopping_patience}'
    if 'max_runtime' in hyperparams:
        max_runtime = hyperparams.pop('max_runtime')
        config['max_runtime'] = max_runtime
        model_name += f'_maxr{max_runtime}'
    if 'mp_ignore_udf' in hyperparams:
        config['mp_ignore_udf'] = hyperparams.pop('mp_ignore_udf')
        model_name = f'ignoreUDF_{model_name}'
    if 'zs_paper_dataset' in hyperparams:
        config['zs_paper_dataset'] = hyperparams.pop('zs_paper_dataset')
        model_name = f'zspaper_{model_name}'
    if 'plans_have_no_udf' in hyperparams:
        config['plans_have_no_udf'] = hyperparams.pop('plans_have_no_udf')
        model_name = f'noUDF_{model_name}'
    if 'train_udf_graph_against_udf_runtime' in hyperparams:
        config['train_udf_graph_against_udf_runtime'] = hyperparams.pop('train_udf_graph_against_udf_runtime')
        model_name = f'udfonly_{model_name}'
    if 'work_with_udf_repr' in hyperparams:
        config['work_with_udf_repr'] = hyperparams.pop('work_with_udf_repr')
        model_name = f'udfrepr_{model_name}'
    if 'pretrained_model_artifact_dir' in hyperparams:
        config['pretrained_model_artifact_dir'] = hyperparams.pop('pretrained_model_artifact_dir')
        model_name = f'pretrained_{model_name}'
    if 'pretrained_model_filename' in hyperparams:
        config['pretrained_model_filename'] = hyperparams.pop('pretrained_model_filename')
        model_name = f'pretrained_{model_name}'
    if 'test_with_count_edges_msg_aggr' in hyperparams:
        config['test_with_count_edges_msg_aggr'] = hyperparams.pop('test_with_count_edges_msg_aggr')
        model_name = f'testctedges_{model_name}'
    if 'min_runtime_ms' in hyperparams:
        tmp = hyperparams.pop('min_runtime_ms')
        if tmp != config['min_runtime_ms']:
            config['min_runtime_ms'] = tmp
            model_name = f'minrt{tmp}ms_{model_name}'
    if 'card_type' in hyperparams:
        card = hyperparams.pop('card_type')
        config['card_type'] = card
        model_name = f'{card}_{model_name}'

    if 'optimizer' in hyperparams:
        optimizer = hyperparams.pop('optimizer')
        if optimizer == 'adam':
            config['optimizer_class_name'] = 'Adam'
            config['optimizer_kwargs'] = dict(lr=0.001)
        elif optimizer == 'sgd':
            config['optimizer_class_name'] = 'SGD'
            config['optimizer_kwargs'] = dict(lr=0.01)
        elif optimizer == 'adamw':
            config['optimizer_class_name'] = 'AdamW'
            config['optimizer_kwargs'] = dict(lr=0.001)
        elif optimizer == 'rmsprop':
            config['optimizer_class_name'] = 'RMSprop'
            config['optimizer_kwargs'] = dict(lr=0.001)
        elif optimizer == 'adagrad':
            config['optimizer_class_name'] = 'Adagrad'
            config['optimizer_kwargs'] = dict(lr=0.001)
        else:
            raise ValueError(f"Unknown optimizer {optimizer}")
        model_name += f'_{optimizer}'

    # extract filter plans from hyperparams
    if 'min_num_branches' in hyperparams:
        min_num_branches = hyperparams.pop('min_num_branches')
        config['filter_plans']['min_num_branches'] = min_num_branches
        model_name += f'_minbranch{min_num_branches}'
    if 'max_num_branches' in hyperparams:
        max_num_branches = hyperparams.pop('max_num_branches')
        config['filter_plans']['max_num_branches'] = max_num_branches
        model_name += f'_maxbranch{max_num_branches}'
    if 'min_num_loops' in hyperparams:
        min_num_loops = hyperparams.pop('min_num_loops')
        config['filter_plans']['min_num_loops'] = min_num_loops
        model_name += f'_minloop{min_num_loops}'
    if 'max_num_loops' in hyperparams:
        max_num_loops = hyperparams.pop('max_num_loops')
        config['filter_plans']['max_num_loops'] = max_num_loops
        model_name += f'_maxloop{max_num_loops}'
    if 'min_num_np_calls' in hyperparams:
        min_num_np_calls = hyperparams.pop('min_num_np_calls')
        config['filter_plans']['min_num_np_calls'] = min_num_np_calls
        model_name += f'_minnp{min_num_np_calls}'
    if 'max_num_np_calls' in hyperparams:
        max_num_np_calls = hyperparams.pop('max_num_np_calls')
        config['filter_plans']['max_num_np_calls'] = max_num_np_calls
        model_name += f'_maxnp{max_num_np_calls}'
    if 'min_num_math_calls' in hyperparams:
        min_num_math_calls = hyperparams.pop('min_num_math_calls')
        config['filter_plans']['min_num_math_calls'] = min_num_math_calls
        model_name += f'_minmath{min_num_math_calls}'
    if 'max_num_math_calls' in hyperparams:
        max_num_math_calls = hyperparams.pop('max_num_math_calls')
        config['filter_plans']['max_num_math_calls'] = max_num_math_calls
        model_name += f'_maxmath{max_num_math_calls}'
    if 'min_num_comp_nodes' in hyperparams:
        min_num_comp_nodes = hyperparams.pop('min_num_comp_nodes')
        config['filter_plans']['min_num_comp_nodes'] = min_num_comp_nodes
        model_name += f'_mincomp{min_num_comp_nodes}'
    if 'max_num_comp_nodes' in hyperparams:
        max_num_comp_nodes = hyperparams.pop('max_num_comp_nodes')
        config['filter_plans']['max_num_comp_nodes'] = max_num_comp_nodes
        model_name += f'_maxcomp{max_num_comp_nodes}'

    base_featurization = None
    # extract base featurization from model config
    for keyword in model_config.split('_'):
        if not keyword in config_keywords:
            raise ValueError(f"Unknown keyword {keyword}")

        if config_keywords[keyword] == 'plan_featurization':
            base_featurization = copy.deepcopy(featurization_dict[keyword])
            break
    print(f'Model config: {model_config}')
    for keyword in model_config.split('_'):
        if keyword in config_keywords:
            if config_keywords[keyword] == 'plan_featurization':
                continue
            elif config_keywords[keyword] == 'loss':
                loss_dict = {'qloss': 'QLoss', 'mse': 'MSELoss', 'procloss': 'ProcentualLoss'}
                final_mlp_kwargs['loss_class_name'] = loss_dict[keyword]
            elif config_keywords[keyword] == 'featurization':
                if keyword == 'liboh':
                    assert 'lib_onehot' not in base_featurization[
                        'COMP_FEATURES'], f'lib_onehot already in COMP_FEATURES {base_featurization["COMP_FEATURES"]}'
                    base_featurization['COMP_FEATURES'].append('lib_onehot')
                elif keyword == 'libemb':
                    assert 'lib_embedding' not in base_featurization[
                        'COMP_FEATURES'], f'lib_embedding already in COMP_FEATURES {base_featurization["COMP_FEATURES"]}'
                    base_featurization['COMP_FEATURES'].append('lib_embedding')
                else:
                    raise ValueError(f"Unknown keyword {keyword}")
            elif config_keywords[keyword] == 'model':
                if keyword == 'gradnorm':
                    config['apply_gradient_norm'] = True
                elif keyword == 'mldupl':
                    config['multi_label_keep_duplicates'] = True
                elif keyword == 'nolrred':
                    config['apply_lr_reduction_on_plateau'] = False
                elif keyword == 'loopend':
                    config['w_loop_end_node'] = True
                elif keyword == 'loopedge':
                    config['add_loop_loopend_edge'] = True
                elif keyword == 'lazy':
                    config['card_est_assume_lazy_eval'] = True
                elif keyword == 'gradntemb':
                    node_type_kwargs['allow_gradients_on_embeddings'] = True
                else:
                    raise ValueError(f"Unknown keyword {keyword}")
        elif keyword.startswith('graphdim'):
            final_mlp_kwargs['input_dim'] = int(keyword[9:])
            tree_layer_kwargs['hidden_dim'] = int(keyword[10:])
            node_type_kwargs['output_dim'] = int(keyword[10:])
        elif keyword.startswith('ntmaxembdim'):
            node_type_kwargs['max_emb_dim'] = int(keyword[11:])
        elif keyword.startswith('dropout'):
            dropout_val = float(keyword[7:])
            if dropout_val > 0:
                fc_out_kwargs['p_dropout'] = dropout_val
                fc_out_kwargs['dropout'] = True
            else:
                fc_out_kwargs['p_dropout'] = 0.0
                fc_out_kwargs['dropout'] = False
        elif keyword.startswith('offsetnp'):
            config['offset_np_import'] = int(keyword[16:])
        else:
            raise ValueError(f"Unknown keyword {keyword}")

    assert base_featurization is not None, f'No base featurization found in model config {model_config}'

    final_mlp_kwargs.update(**fc_out_kwargs)
    tree_layer_kwargs.update(**fc_out_kwargs)
    node_type_kwargs.update(**fc_out_kwargs)

    config['final_mlp_kwargs'] = final_mlp_kwargs
    config['tree_layer_kwargs'] = tree_layer_kwargs
    config['node_type_kwargs'] = node_type_kwargs
    config['featurization'] = base_featurization

    assert len(hyperparams) == 0, f'Not all hyperparams were used: {hyperparams}'

    return config, train_wl_paths, test_wl_paths, statistics_file, model_name


def compile_train_test_filenames(hyperparams: Dict, wl_base_path: str, zs_paper_dataset: bool = False,
                                 include_no_udf_data: bool = False, include_no_udf_data_large: bool = False,
                                 include_pushdown_data: bool = True, include_pullup_data: bool = False,
                                 include_select_only_w_branch: bool = False):
    # data
    data_keyword = hyperparams.pop('data_keyword')
    workload_filename = 'workload.json'
    if data_keyword == 'complex_dd':
        data_dir = 'duckdb_pushdown'
        dataset_list_name = 'zs_less_scaled'
    else:
        raise ValueError(f"Unknown data keyword {data_keyword}")

    # test against which dataset (zero-shot)
    if 'test_against' in hyperparams:
        test_against = hyperparams.pop('test_against')
    else:
        test_against = 'imdb'

    model_name_suffix = f'{test_against}'

    # train on test dataset
    if 'train_on_test' in hyperparams:
        train_on_test = hyperparams.pop('train_on_test')
        assert train_on_test, train_on_test
        model_name_suffix += '_train_on_test'
    else:
        train_on_test = False

    # compile workload runs / test workload runs
    train_wl_paths = []
    test_wl_paths = []

    base_path = os.path.join(wl_base_path, data_dir)

    for dataset in [dataset for dataset in dataset_list_dict[dataset_list_name]]:
        path_list = []

        if include_pushdown_data:
            path = os.path.join(base_path, 'parsed_plans', dataset.db_name, workload_filename)
            if os.path.exists(path):
                path_list.append(path)
            else:
                print(f'No {data_keyword} data found for {dataset.db_name} ({path})')

        if include_pullup_data:
            path = os.path.join(
                wl_base_path, 'duckdb_pullup', 'parsed_plans',
                dataset.db_name, 'workload.json')
            if os.path.exists(path):
                path_list.append(path)
            else:
                print(f'No pullup data found for {dataset.db_name}')

        if include_no_udf_data:
            path = os.path.join(
                wl_base_path, 'duckdb_no_udf', 'parsed_plans',
                dataset.db_name, 'workload.json')
            if os.path.exists(path):
                path_list.append(path)
            else:
                print(f'No no-udf data found for {dataset.db_name}')

        if include_select_only_w_branch:
            path = os.path.join(
                wl_base_path, 'duckdb_scan_only', 'parsed_plans',
                dataset.db_name, 'workload.json')
            if os.path.exists(path):
                path_list.append(path)
            else:
                print(f'No select only with branch data found for {dataset.db_name}')

        if dataset.db_name == test_against:
            test_wl_paths.extend(path_list)

            if train_on_test:
                train_wl_paths.extend(path_list)
        elif train_on_test:
            # skip this dataset since we are only training on test db
            continue
        else:
            train_wl_paths.extend(path_list)

    statistics_file = os.path.join(base_path, 'parsed_plans', 'statistics_workload_combined.json')

    assert len(test_wl_paths) > 0, f'No test workloads found for {data_keyword}'

    return train_wl_paths, test_wl_paths, statistics_file, data_keyword, model_name_suffix
