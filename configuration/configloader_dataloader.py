from configuration.configloader_base import read_config, convert_key_to_int, get_key_list

from configuration.keys import DataLoaderKeys as DLK

SET_PARAMS = {DLK.LD_MASK: DLK.MASK_COLOR, DLK.LD_LABEL: DLK.TISSUE_LABELS, DLK.LD_COLOR: DLK.PLOT_COLORS}


def read_dataloader_config(file: str, section: str):
    """Reads a JSOn file and returns a dictionary with the configuration for the dataloader.
    It converts the Label number from a string to a number and split
    :param file: JSON file to load
    :param section: Section to read

    :return: Dictionary with configurations for dataloader
    """
    dataloader = read_config(file=file, section=section)
    dataloader[DLK.LABEL_DATA] = convert_key_to_int(str_dict=dataloader[DLK.LABEL_DATA])
    dataloader[DLK.LABELS] = get_key_list(dataloader[DLK.LABEL_DATA])
    label_data_split = split_label_data(label_data=dataloader[DLK.LABEL_DATA])
    dataloader = set_parameter(configuration=dataloader, params=label_data_split, replace_dict=SET_PARAMS,
                               origin_label=DLK.LABEL_DATA)

    dataloader.pop(DLK.LABEL_DATA)

    return dataloader


def split_label_data(label_data: dict) -> dict:
    """Change the dictionary from classification orientation to an orientation to the sub values.

    :param label_data: Data to change

    :return: Sub value orientated dictionary

    Example
    -------
    >>> DATA = {0: {"sub_key_0": [0, 1], "sub_key_1": 2}, 1: {"sub_key_0": [3, 4], "sub_key_1": 5}}
    >>> split_label_data(DATA)
    >>> {"sub_key_0": {0: [0, 1], 1: [3, 4]}, "sub_key_1": {0: 2, 1: 5}}
    """
    data_per_parm = {}
    for sub_key in list(label_data.values())[0].keys():
        data_per_parm[sub_key] = {}

    for classification, sub_values in label_data.items():
        for param, value in sub_values.items():
            data_per_parm[param].update({classification: value})

    return data_per_parm


def set_parameter(configuration: dict, params: dict, replace_dict: dict, origin_label: str) -> dict:
    """Add keys and values to a dictionary and replace the key when there is another given key.

    :param configuration: Dictionary where the keys and value to add
    :param params: Keys and values to add
    :param replace_dict: Keys to replace. Key -> original key, value -> new key
    :param origin_label: Name from replace_dict

    :return: New dictionary with added keys and values

    :raises ValueError: If a key is already in base dictionary

    Example
    -------
    >>> CONF = {"key_0": 0}
    >>> PARAMS = {"sub_key_0": 1, "sub_key_1": 2, "sub_key_2": 3}
    >>> REPLACE = {"sub_key_0": "key_1", "sub_key_1": "key_2"}
    >>> set_parameter(CONF, PARAMS, REPLACE, "test")
    >>> {"key_0": 0, "key_1": 1, "key_2": 2, "sub_key_2": 3}
    """
    config = configuration.copy()
    for k, v in params.items():
        if k in replace_dict.keys():
            config[replace_dict[k]] = v
        else:
            if k not in config.keys():
                config[k] = v
            else:
                raise ValueError(f"Key '{k}' from {origin_label} is already in dataloader configs. Please rename key!")

    return config
