import json
import os
from types import SimpleNamespace

from pkg_resources import resource_filename


def load_schema_json(dataset: str):
    schema_path = resource_filename(__name__, os.path.join(dataset, 'schema.json'))
    assert os.path.exists(schema_path), f"Could not find schema.json ({schema_path})"
    schema = load_json(schema_path)

    # apply schema modifications
    if dataset == 'financial':
        assert schema.tables[2] == 'order'
        schema.tables[2] = 'orders'

        assert schema.relationships[5][0] == 'order'
        schema.relationships[5][0] = 'orders'

    return schema


def load_json(path, namespace=True):
    with open(path) as json_file:
        if namespace:
            json_obj = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        else:
            json_obj = json.load(json_file)
    return json_obj
