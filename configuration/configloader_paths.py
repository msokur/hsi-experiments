import os
import platform

from configuration.configloader_base import read_config, concat_dict

from configuration.keys import PathKeys as PK


CONCAT_KEY = PK.CONCAT_WITH


def read_path_config(file: str, system_mode: str, database: str) -> dict:
    """Reads a JSON file and returns a dictionary with paths.
    The paths in the 'system_mode' section are OS specific and the paths in the 'database' section can be prefixed and
    must be a list with separate strings for every folder name.

    :param file: JSON file to read
    :param system_mode: OS specific section
    :param database: Database  structure

    :return: Dictionary with paths
    """
    data_system = read_config(file=file, section=system_mode)
    data_database = read_config(file=file, section=database)
    data_base = read_config(file=file, section="BASE_PATH")

    path_dict = concat_dict(data_system, data_base)
    path_dict = get_database_paths(path_dict=path_dict, to_prefix=data_database)
    if platform.system() == 'Windows':
        path_dict[PK.SYS_DELIMITER] = "\\"
    else:
        path_dict[PK.SYS_DELIMITER] = "/"

    return path_dict


def get_database_paths(path_dict: dict, to_prefix: dict) -> dict:
    """Concatenate two dictionary and set prefixes.
    If 'CONCAT_WITH_' in the key the path will be prefixed with key after the word 'CONCAT_WITH_'.
    The path from the 'to_prefix' parameter must be a list with separate strings for every folder name.

    :param path_dict: First dictionary with prefix paths
    :param to_prefix: Dictionary with paths to prefix

    :return: A dictionary with all paths from the two dictionary's

    :raises ValueError: If the 'CONCAT_WITH_' in the wrong order.

    Example
    -------
    >>> PATH_DICT = {"path_0": "folder_0/folder_1", "path_1": "folder_X/folder_Y"}
    >>> TO_PREFIX = {"CONCAT_WITH_path_0": {"val_0": ["x", "y"], "val_1": ["3"]},
    >>>              "CONCAT_WITH_val_1": {"val_3": ["1", "2"]}}
    >>> get_database_paths(PATH_DICT, TO_PREFIX)
    >>> {"path_0": "folder_0/folder_1", "path_1": "folder_X/folder_Y", "val_0": "folder_0/folder_1/x/y",
    >>>  "val_1": "folder_0/folder_1/3", "val_3": "folder_0/folder_1/3/1/2"}
    """
    path_dict_ = path_dict.copy()
    for key, values in to_prefix.items():
        if CONCAT_KEY in key:
            prefix = key.split(CONCAT_KEY)[1]
            if prefix in path_dict_.keys():
                temp = set_prefix(prefix=path_dict_[prefix], database=values)
                path_dict_ = concat_dict(path_dict_, temp)
            else:
                raise ValueError(f"Check the order form your Database paths, prefix: '{prefix}' not found!")
        else:
            path_dict_[key] = os.path.join(*values)

    return path_dict_


def set_prefix(prefix: str, database: dict) -> dict:
    """Set a prefix for every path in the database dictionary.
    The path must be a list with separate strings for every folder name.

    :param prefix: String with prefix
    :param database: To prefix paths.

    :return: Dictionary with the prefixed paths.

    Example
    -------
    >>> PREFIX = "folder_0"
    >>> DATABASE = {"val_0": ["folder_1", "folder_2"], "val_1": ["folder_3"]}
    >>> set_prefix(PREFIX, DATABASE)
    {"val_0": "folder_0/folder_1/folder_2", "val_1": "folder_0/folder_3"}
    """
    prefixed_paths = {}
    for key, value in database.items():
        prefixed_paths[key] = os.path.join(prefix, *value)

    return prefixed_paths
