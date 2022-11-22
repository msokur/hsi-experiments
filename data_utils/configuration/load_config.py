import json
import platform
import os


def read_path_config(file: str, system_mode: str, database: str) -> dict:
    data_system = read_config(file=file, section=system_mode)
    data_database = read_config(file=file, section=database)
    data_base = read_config(file=file, section="BASE_PATH")

    data = {}
    if platform.system() == 'Windows':
        data["SYSTEM_PATHS_DELIMITER"] = "\\"
    else:
        data["SYSTEM_PATHS_DELIMITER"] = "/"

    data = concat_dict(data_system, data_base)
    prefixed = set_prefix(data_system["PREFIX"], data_database)
    data = concat_dict(data, prefixed)
    return data


def set_prefix(prefix: str, database: dict) -> dict:
    prefixed_paths = {}
    for key, value in database.items():
        if key not in ["SHUFFLED_PATH", "BATCHED_PATH"]:
            prefixed_paths[key] = os.path.join(prefix, *value)

    prefixed_paths["SHUFFLED_PATH"] = os.path.join(prefixed_paths["RAW_NPZ_PATH"], *database["SHUFFLED_PATH"])
    prefixed_paths["BATCHED_PATH"] = os.path.join(prefixed_paths["RAW_NPZ_PATH"], *database["BATCHED_PATH"])
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
