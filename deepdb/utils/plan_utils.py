import csv
import os
from enum import Enum


def dumper(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, Enum):
        return str(obj)
    try:
        return obj.toJSON()
    except:
        try:
            return obj.__dict__
        except Exception as e:
            print(f'Obj: {type(obj)} / {obj}')
            raise e

def save_csv(csv_rows, target_csv_path):
    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)

    # make sure the first row contains all possible keys. Otherwise dictwriter raises an error.
    for csv_row in csv_rows:
        for key in csv_row.keys():
            if key not in csv_rows[0].keys():
                csv_rows[0][key] = None

    with open(target_csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, csv_rows[0].keys())
        for i, row in enumerate(csv_rows):
            if i == 0:
                w.writeheader()
            w.writerow(row)