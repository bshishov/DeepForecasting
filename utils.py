import json
import importlib
import csv
import os


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def instantiate(path: str, *args, **kwargs):
    if not path:
        return None

    # If path is a path to json config file
    # Then instantiate from json
    if path.endswith('.json'):
        with open(path) as f:
            config = json.loads(f.read())  # type: dict
            class_name = config.pop('class')  # type: str

            # Update the kwargs from config (rewriting function kwargs)
            kwargs.update(config)
    else:
        # Otherwise instantiate assuming that path is a path to the class
        class_name = path

    parts = class_name.split('.')
    if len(parts) == 1:
        cls_obj = class_for_name(None, parts[-1])
    else:
        cls_obj = class_for_name('.'.join(parts[:-1]), parts[-1])

    return cls_obj(*args, **kwargs)


def write_row_to_csv(csv_path, **kwargs):
    exists = os.path.exists(csv_path)
    if not exists and not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))
    field_names = sorted(list(kwargs.keys()))
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if not exists:
            writer.writeheader()
        writer.writerow(kwargs)
