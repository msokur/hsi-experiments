from typing import List, Union
import warnings
import numpy as np
import os

from configuration.keys import DataLoaderKeys as DLK


def get_splits(typ: str, paths: list, values: Union[int, list[list[str]], tuple]) -> List[np.ndarray]:
    """
    Split the paths ny a type.

    :param typ: Type of split.
    :param paths: Paths to split.
    :param values: The value to split the paths.

    :return: A list with numpy arrays with the split integers.

    :raises UserWarning: For wrong value typ or for a wrong split typ.
    """
    if typ == "Name":
        if isinstance(values, tuple):
            return split_name(paths, values)
        else:
            __warning(DLK.PATIENTS_EXCLUDE_FOR_TEST)
    elif typ == "List":
        if isinstance(values, list):
            return split_list(paths, values)
        else:
            __warning(DLK.PATIENTS_EXCLUDE_FOR_TEST)
    elif typ == "Files":
        if isinstance(values, int):
            return split_int(len(paths), values)
        else:
            __warning(DLK.PATIENTS_EXCLUDE_FOR_TEST)
    else:
        __warning(DLK.SPLIT_PATHS_BY)

    return split_int(len(paths), 1)


def split_name(paths: list, name_slice: tuple) -> List[np.ndarray]:
    """
    Split and group the indexes from the paths list by given slice for the file name. It will be use the file name
    without the file extension to get the indexes.

    :param paths: List with the paths to split and group.
    :param name_slice: Tuple with the indices to start and stop (not included).

    :return: A list with the grouped indexes from the paths list.

    Example
    -------
    >>> path_list = ['/dir/data0_1.dat', '/dir/data1_1.dat', '/dir/data0_2.dat', '/dir/data1_2.dat']
    >>> names_slice = (0, 5)
    >>> split_name(path_list, names_slice)
    [array([0, 2]), array([1, 3])]

    >>> path_list = ['/dir/data0_1.dat', '/dir/data1_1.dat', '/dir/data0_2.dat', '/dir/data1_2.dat']
    >>> names_slice = (0, 7)
    >>> split_name(path_list, names_slice)
    [array([0]), array([2]), array([1]), array([3])]
    """
    unique = np.unique([os.path.split(path)[-1].split(".")[0][slice(name_slice[0], name_slice[1])] for path in paths])
    splits = []
    for name in unique:
        splits.append(np.flatnonzero([True if name in path else False for path in paths]))

    return splits


def split_list(paths: list[str], name_list: list[list[str]]) -> List[np.ndarray]:
    """
    Split and group the indexes from the paths list by a name list.

    :param paths: List with paths to group and split.
    :param name_list: A List with names to group.

    :return: A list with the grouped indexes from the paths list.

    :raises UserWarning: No match with a name from name_list in the path list.
    :raises ValueError: More than one name found for one name.

    Example
    -------
    >>> path_list = ['/dir/data0', '/dir/data1', '/dir/data2', '/dir/data3']
    >>> names_list = [['data0', 'data1'], ['data2'], ['data3']]
    >>> split_list(path_list, names_list)
    [array([0, 1]), array([2]), array([3])]
    """
    splits = []
    for names in name_list:
        split_part_list = []
        for name in names:
            name_idx = np.flatnonzero([True if name == os.path.splitext(os.path.basename(path))[0] else False for path
                                       in paths])
            if len(name_idx) < 1:
                warnings.warn(f"No matches with name '{name}' in paths. The name will be ignored!")
                continue
            elif len(name_idx) > 1:
                raise ValueError(f"More than one match with the name '{name}'. Check your split name list!")
            else:
                split_part_list.append(name_idx[0])
        if len(split_part_list) > 0:
            splits.append(np.array(split_part_list, dtype=np.uint8))

    return splits


def split_int(path_length: int, num: int) -> List[np.ndarray]:
    """
    Splits a range of integers in equally parts. The range has the length from 'path_length' and in every part there are
    min. the number of 'num' values.

    If the path length is x and the num y, the first array have x//y + 1 values and the rest x//y.

    :param path_length: A integer with the range to split.
    :param num: Number of values per split.

    :return: A list with numpy arrays with the split integers.

    :raises ValueError: When path_length < num.

    Example
    -------
    >>> x = 4
    >>> y = 2
    >>> z = split_int(x, y)
    [array([0, 1]), array([2, 4])]

    >>> x = 5
    >>> y = 2
    >>> z = split_int(x, y)
    [array([0, 1, 2]), array([3, 4])]
    """
    return np.array_split(range(path_length), path_length // num)


def __warning(msg: str):
    warnings.warn(f"'{msg}' in configurations is not correct. Paths will be split by files with 1!")
