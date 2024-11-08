import collections
import os

from pkg_resources import resource_filename

from deepdb.datasets.column_types import Datatype
from deepdb.datasets.utils import load_schema_json, load_json
from deepdb.ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_schema(orig_dataset, source_dir, csv_read_kwargs=None):
    # remove _scaled123 suffix
    if '_scaled' in orig_dataset:
        dataset = orig_dataset.split('_scaled')[0]
    else:
        dataset = orig_dataset

    schema = load_schema_json(dataset)
    col_stats = load_json(resource_filename(__name__, f'{dataset}/column_statistics.json'), namespace=False)

    if csv_read_kwargs is None:
        custom_nan_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND',
                             '1.#QNAN', '<NA>', 'N/A', 'NULL', 'NaN', 'n/a', 'nan', 'null']
        read_kwargs = dict(**vars(schema.csv_kwargs), keep_default_na=False, na_values=custom_nan_values)
    else:
        read_kwargs = csv_read_kwargs

    print(f'Read {orig_dataset} with {read_kwargs}', flush=True)

    schema_graph = SchemaGraph()

    join_keys = collections.defaultdict(set)
    scaled_join_keys = collections.defaultdict(set)
    compound_keys = collections.defaultdict(dict)
    relationships = []
    table_pairs = set()
    for t_l, c_l, t_r, c_r in schema.relationships:
        if (t_l, t_r) in table_pairs or (t_r, t_l) in table_pairs:
            print(f"Warning: duplicate FK between {(t_l, t_r)}")
            continue

        c_l_compound = c_l
        c_r_compound = c_r

        if isinstance(c_l, list) or isinstance(c_l, tuple):
            scaled_join_keys[t_l].update(c_l)
            scaled_join_keys[t_r].update(c_r)

            if len(c_l) == 1:
                c_l_compound = c_l[0]
                c_r_compound = c_r[0]
            else:
                c_l_compound = '_'.join(c_l)
                c_r_compound = '_'.join(c_r)
                compound_keys[t_l][c_l_compound] = c_l
                compound_keys[t_r][c_r_compound] = c_r
        else:
            scaled_join_keys[t_l].add(c_l)
            scaled_join_keys[t_r].add(c_r)

        join_keys[t_l].add(c_l_compound)
        join_keys[t_r].add(c_r_compound)
        table_pairs.add((t_l, t_r))
        relationships.append((t_l, c_l_compound, t_r, c_r_compound))

    for t in schema.tables:

        if source_dir is not None:
            if dataset == 'financial' and t == 'orders':
                source_t_path = os.path.join(source_dir, f'order.csv')
                if not os.path.exists(source_t_path):
                    source_t_path = os.path.join(source_dir, f'{t}.csv')
            else:
                source_t_path = os.path.join(source_dir, f'{t}.csv')
        else:
            source_t_path = None
        #
        # df_header = pd.read_csv(source_t_path, nrows=1, **read_kwargs)

        irrelevant_attributes = []
        curr_col_stats = col_stats.get(t)
        assert curr_col_stats is not None, f'No column statistics for table {t} found in {col_stats.keys()}.'

        for c, vals in curr_col_stats.items():
            if vals['datatype'] == str(Datatype.MISC):
                if c not in join_keys[t]:
                    irrelevant_attributes.append(c)

        # attributes_df = list(df_header.columns)
        attributes = list(curr_col_stats.keys())

        # assert attributes_df == attributes, f'Attributes in schema and csv file do not match for table {t}: {attributes_df} vs {attributes}'

        no_compression = [a for a in attributes if a not in irrelevant_attributes and (a in scaled_join_keys[t] or
                                                                                       a in {'kind_id', 'info_type_id',
                                                                                             'role_id', 'keyword_id',
                                                                                             'company_id',
                                                                                             'company_type_id'})]

        if len(irrelevant_attributes) > 0:
            print(f'irrelevant_attributes ({t}): {irrelevant_attributes}', flush=True)
        schema_graph.add_table(Table(t, attributes=attributes,
                                     irrelevant_attributes=irrelevant_attributes,
                                     no_compression=no_compression,
                                     csv_file_location=source_t_path,
                                     primary_key=None,
                                     compound_keys=compound_keys.get(t),
                                     table_size=2))

    for t_l, c_l, t_r, c_r in relationships:
        schema_graph.add_relationship(t_l, c_l, t_r, c_r)

    schema_graph.read_kwargs = read_kwargs
    schema_graph.join_keys = join_keys
    schema_graph.scaled_join_keys = scaled_join_keys
    # print(schema_graph.scaled_join_keys)

    return schema_graph
