import json


CONVERT_KEY = ["MASK_COLOR", "TISSUE_LABELS", "PLOT_COLORS"]


def concat_dict(dict1: dict, dict2: dict) -> dict:
    """Concatenate dictionary

    Concatenate two dictionary a returns a new one with both datas.

    :param dict1: First dictionary.
    :param dict2: Second dictionary.

    :returns: Combined dictionary.

    :raises ValueError: If one key is in both dictionary.
    """
    dict_temp = dict1.copy()
    for key, value in dict2.items():
        if key not in dict1:
            dict_temp[key] = value
        else:
            raise ValueError(f'The key {key} is already in the Dictionary!')
    return dict_temp


def convert_key_to_int(str_dict: dict):
    """Convert keys to integer

    Converting string dictionary keys to integer keys if it is possible.

    :param str_dict: Dictionary with number as string key.

    :returns: Dictionary withe integer keys.

    Example
    -------
    >>> a = {"0": "dict_value_0", "1": "dict_value_1"}
    >>> convert_key_to_int(str_dict=a)
    ... {0: "dict_value_0", 1: "dict_value_1"}
    """
    int_dict = {}
    for key, value in str_dict.items():
        int_dict[int(key)] = value
    return int_dict


def get_key_list(dict_data: dict) -> list:
    """
    Returns a list with all keys in the dictionary.

    :param dict_data: Dictionary with the keys.

    :returns: List with keys from the dictionary.

    Example
    -------
    >>> a = {0: "value_0", 1: "value_1"}
    >>> get_key_list(dict_data=a)
    array([0, 1])
    >>> a = {"zero": "value_zero", "one": "value_one"}
    >>> get_key_list(dict_data=a)
    array(["zero", "one"])
    """
    return [*dict_data]


def read_config(file: str, section: str) -> dict:
    """Read configuration file

    Reads a section in a json-file and returns a dictionary with the parameter.

    :param file: File path.
    :param section: Section name to read.

    :returns: Dictionary with parameter.

    :raises ValueError: If no section found with the given section name.
    """
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
