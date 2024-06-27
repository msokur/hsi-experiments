import os
from typing import List, Dict


def get_sort(paths: list, number: bool, split: list) -> list:
    """
    Sort paths by names or by number. To use sort by number set number to 'True' and write the string before and after
    the number in the list 'split'. By sort with number you can only detect a number in the file name.

    :param paths: A list withs paths to sort.
    :param number: Boolean value, True when sort paths by number.
    :param split: A list with the letters before and after the number.

    return: A list with sorted paths.
    """
    if number:
        return sort(paths=paths, split=split)
    else:
        return sorted(paths)


def sort(paths: list, split: list) -> list:
    """
    Sort the paths by number in the file name that is between the two strings in split.

    :param paths: A list with the paths to sort
    :param split: A list with the letters before and after the number.

    :return: A sorted list with paths by numbers.

    Example
    -------
    >>> a = ['C:/folder2/data1cube.dat', 'C:/folder3/data2cube.dat', 'C:/folder4/data0cube.dat']
    >>> b = ['data', 'cube']
    >>> c = sort(a, b)
    ['C:/folder4/data0cube.dat', 'C:/folder2/data1cube.dat', 'C:/folder3/data2cube.dat']
    """

    def take_only_number(elem):
        return get_number(elem=elem, split=split)

    paths = sorted(paths, key=take_only_number)

    return paths


def get_number(elem: str, split: list) -> int:
    """
    Get the number (integer) from a string. The split parameter is a list with the letters before and after the number.
    Is the number at the beginning from the string, write at first 'None', is the number at the end of the string write
    'None' in the second insert.
    Is the string a path, the last part will be used.

    :param elem: The string with the integer.
    :param split: A list with the letters before and after the number.

    :return: The integer from the string.

    :raise ValueError: If the splits Value are incorrect and the integer can't be found.
    :raise TypeError: If the splits are a 'None Type'.

    Example
    -------
    >>> a = 'C:/folder/data0cube.dat'
    >>> b = ['data', 'cube']
    >>> c = get_number(a, b)
    0

    >>> a = 'C:/folder/1cube.dat'
    >>> b = [None, 'cube']
    >>> c = get_number(a, b)
    1

    >>> a = 'C:/folder/data2.dat'
    >>> b = ['data', None]
    >>> c = get_number(a, b)
    2
    """
    elem = os.path.split(elem)[-1]
    if "\\" in elem:
        elem = elem.split("\\")[-1]
    try:
        number = int(elem.split(split[0])[-1].split(".")[0].split(split[1])[0])
    except ValueError:
        raise ValueError("Can't change literal to integer. Check your 'splits' parameter!")
    return number


def folder_sort(paths: List[str], depth: int = 1) -> Dict[str, List[str]]:
    if depth < 1:
        raise ValueError("Depth smaller then 1 is not allowed by folder sort!")

    names_and_paths = {}
    for p in paths:
        if "\\" in p:
            folder = p.replace("\\", "/")
        else:
            folder = p
        folder = os.path.split(p=folder)[0]
        for i in range(depth - 1):
            folder = os.path.split(p=folder)[0]
        folder_name = os.path.split(p=folder)[-1]

        if folder_name not in names_and_paths:
            names_and_paths[folder_name] = [p]
        else:
            names_and_paths[folder_name].append(p)

    for name in names_and_paths.keys():
        sorted(names_and_paths[name])

    return names_and_paths
