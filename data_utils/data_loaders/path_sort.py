import os


def get_sort(paths: list, number: bool, splits: list) -> list:
    if number:
        return sort(paths, splits)
    else:
        return sorted(paths)


def get_number(elem: str, splits: list) -> str:
    elem = os.path.split(elem)[-1]
    return elem.split(splits[0])[-1].split(".")[0].split(splits[1])[0]


def sort(paths: list, split: list) -> list:
    def take_only_number(elem):
        return int(get_number(elem=elem, splits=split))

    paths = sorted(paths, key=take_only_number)

    return paths
