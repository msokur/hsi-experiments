import json


CONVERT_KEY = ["MASK_COLOR", "TISSUE_LABELS", "PLOT_COLORS"]


def concat_dict(dict1: dict, dict2: dict) -> dict:
    dict_temp = dict1.copy()
    for key, value in dict2.items():
        if key not in dict1:
            dict_temp[key] = value
        else:
            raise ValueError(f'The key {key} is already in the Dictionary!')
    return dict_temp


def convert_key_to_int(str_dict: dict):
    int_dict = {}
    for key, value in str_dict.items():
        int_dict[int(key)] = value
    return int_dict


def read_config(file: str, section: str) -> dict:
    with open(file, "r") as config_file:
        data = json.load(config_file)

    if section in data:
        data_ = data[section]
        for convert in CONVERT_KEY:
            if convert in data_:
                data_[convert] = convert_key_to_int(data_[convert])
        return data_
    else:
        raise ValueError(f'Section {section}, not found in the {file} file!')
