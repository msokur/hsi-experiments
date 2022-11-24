import json
import platform
import os

CONCAT_KEY = "CONCAT_WITH_"


def read_path_config(file: str, system_mode: str, database: str) -> dict:
    data_system = read_config(file=file, section=system_mode)
    data_database = read_config(file=file, section=database)
    data_base = read_config(file=file, section="BASE_PATH")

    path_dict = concat_dict(data_system, data_base)
    path_dict = get_database_paths(path_dict=path_dict, to_prefix=data_database)
    if platform.system() == 'Windows':
        path_dict["SYSTEM_PATHS_DELIMITER"] = "\\"
    else:
        path_dict["SYSTEM_PATHS_DELIMITER"] = "/"

    return path_dict


def get_database_paths(path_dict: dict, to_prefix: dict) -> dict:
    path_dict_ = path_dict.copy()
    for key, values in to_prefix.items():
        if CONCAT_KEY in key:
            prefix = key.split(CONCAT_KEY)[1]
            if prefix in path_dict_.keys():
                temp = set_prefix(prefix=path_dict_[prefix], database=values)
                path_dict_ = concat_dict(path_dict_, temp)
            else:
                raise ValueError(f"Check the order form your Database paths, prefix: {prefix} not found!")
        else:
            path_dict_[key] = os.path.join(*values)

    return path_dict_


def set_prefix(prefix: str, database: dict) -> dict:
    prefixed_paths = {}
    for key, value in database.items():
        prefixed_paths[key] = os.path.join(prefix, *value)

    return prefixed_paths


def concat_dict(dict1: dict, dict2: dict) -> dict:
    dict_temp = dict1.copy()
    for key, value in dict2.items():
        if key not in dict1:
            dict_temp[key] = value
        else:
            raise ValueError(f'The key {key} is already in the Dictionary!')
    return dict_temp


def read_config(file: str, section: str) -> dict:
    with open(file, "r") as config_file:
        data = json.load(config_file)

    if section in data:
        return data[section]
    else:
        raise ValueError(f'Section {section}, not found in the {file} file!')
