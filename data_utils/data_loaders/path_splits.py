import numpy as np


def get_splits(typ: str, paths: list, values, delimiter: str):
    if typ == "Name":
        splits = split_name(paths, values, delimiter)
    elif typ == "List":
        splits = split_list(paths, values)
    else:
        num = values
        if typ != "Files":
            print("SPLIT_BY is not correct. Paths will be split by files with 1!")
            num = 1
        splits = split_int(len(paths), num)

    return splits


def split_int(path_length: int, num: int):
    cv_split = int(path_length / num)
    return np.array_split(range(path_length), cv_split)


def split_list(paths: list, name_list: list):
    paths = np.array(paths)

    splits = []
    for splits in name_list:
        split_part_list = []
        for split in splits:
            split_part_list.append(np.flatnonzero([True if split in path else False for path in paths])[0])
        split_part_list_np = np.array(split_part_list, dtype=np.uint8)
        splits.append(split_part_list_np)

    splits = np.array(splits)

    return np.array_split(np.array(splits), splits.shape[0])


def split_name(paths: list, name_slice: tuple, delimiter: str):
    paths = np.array(paths)

    unique_names = np.unique([path.split(delimiter)[-1][slice(name_slice[0], name_slice[1])] for path in paths],
                             return_index=True)
    splits_ = unique_names[-1]
    splits = [np.arange(splits_[idx], splits_[idx + 1]) if idx < splits_.shape[0] - 1
              else np.arange(splits_[idx], splits_.shape[0]) for idx in range(splits_.shape[0])]

    return np.array(splits)
