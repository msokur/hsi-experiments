from typing import Tuple

import pytest
import os
import numpy as np
from glob import glob

from data_utils.dataset.choice_names import ChoiceNames
from data_utils.data_storage import DataStorageNPZ, DataStorageZARR


def get_paths(data_dir: str, storage_typ: str, not_use: list) -> Tuple[list, list]:
    selected_paths = glob(os.path.join(data_dir, storage_typ + "_file", "choice_data", "choice_test_*"))
    selected_paths = sorted(selected_paths)
    not_use_path = [selected_paths[idx] for idx in not_use]

    return selected_paths, not_use_path


def get_data_storage(storage_typ: str):
    if storage_typ == "zarr":
        return DataStorageZARR()
    else:
        return DataStorageNPZ()


def get_random_choice(data_dir: str, storage_typ: str, not_use: list, size: int) -> Tuple[np.ndarray, list]:
    selected_paths, not_use_path = get_paths(data_dir=data_dir, storage_typ=storage_typ, not_use=not_use)
    choice_names = ChoiceNames(data_storage=get_data_storage(storage_typ=storage_typ), config_cv={}, labels=[],
                               y_dict_name="y")
    paths = choice_names.random_choice(paths=selected_paths, excepts=not_use_path, size=size)

    return paths, not_use_path


RANDOM_CHOICE_DATA = [("npz", [0, 3], 1), ("npz", [0, 1, 2, 3], 3),
                      ("zarr", [6], 9), ("zarr", [], 10)]


@pytest.mark.parametrize("storage_typ,not_use,size", RANDOM_CHOICE_DATA)
def test_random_choice_size(data_dir: str, storage_typ, not_use: list, size: int):
    paths, _ = get_random_choice(data_dir=data_dir, storage_typ=storage_typ, not_use=not_use, size=size)

    assert len(np.unique(paths)) == size


@pytest.mark.parametrize("storage_typ,not_use,size", RANDOM_CHOICE_DATA)
def test_random_choice_duplicates(data_dir: str, storage_typ, not_use: list, size: int):
    paths, not_use_path = get_random_choice(data_dir=data_dir, storage_typ=storage_typ, not_use=not_use, size=size)

    assert not (np.isin(paths, not_use_path)).all()


RANDOM_CHOICE_ERROR_DATA = [("npz", [0, 3], 9), ("npz", [0, 1, 2, 3], 7),
                            ("zarr", [6], 10), ("zarr", [], 11)]


@pytest.mark.parametrize("storage_typ,not_use,size", RANDOM_CHOICE_ERROR_DATA)
def test_random_choice_error(data_dir: str, storage_typ, not_use: list, size: int):
    with pytest.raises(ValueError, match="Cannot take a larger sample than population when 'replace=False'"):
        get_random_choice(data_dir=data_dir, storage_typ=storage_typ, not_use=not_use, size=size)


CLASS_CHOICE_DATA = [("npz", [0, 3], [0, 1, 2, 3, 4]), ("npz", [0, 5, 7], [0, 1, 2]),
                     ("zarr", [6], [0, 1, 2, 3, 4]), ("zarr", [], [0, 1, 4])]


@pytest.mark.parametrize("storage_typ,not_use,labels", CLASS_CHOICE_DATA)
def test_class_choice_labels(data_dir: str, storage_typ, not_use: list, labels: list):
    data_storage = get_data_storage(storage_typ=storage_typ)
    choice_names = ChoiceNames(data_storage=data_storage, config_cv={}, labels=labels, y_dict_name="y")
    paths, not_use_path = get_paths(data_dir=data_dir, storage_typ=storage_typ, not_use=not_use)
    not_use_name = [data_storage.get_name(path=p) for p in not_use_path]
    chosen_names = choice_names.class_choice(paths=paths, paths_names=[data_storage.get_name(path=p) for p in paths],
                                             excepts=not_use_name)
    chosen_labels = []
    for chosen_name in chosen_names:
        data_labels = data_storage.get_data(
            data_path=os.path.join(data_dir, storage_typ + "_file", "choice_data", chosen_name),
            data_name="y")
        chosen_labels += list(data_labels)
    print(chosen_labels)
    assert (np.isin(labels, np.unique(chosen_labels))).all()


@pytest.mark.parametrize("storage_typ,not_use,labels", CLASS_CHOICE_DATA)
def test_class_choice_duplicates(data_dir: str, storage_typ, not_use: list, labels: list):
    data_storage = get_data_storage(storage_typ=storage_typ)
    choice_names = ChoiceNames(data_storage=data_storage, config_cv={}, labels=labels, y_dict_name="y")
    paths, not_use_path = get_paths(data_dir=data_dir, storage_typ=storage_typ, not_use=not_use)
    not_use_name = [data_storage.get_name(path=p) for p in not_use_path]
    chosen_names = choice_names.class_choice(paths=paths, paths_names=[data_storage.get_name(path=p) for p in paths],
                                             excepts=not_use_name)

    assert not (np.isin(chosen_names, not_use_name)).all()


ClASS_CHOICE_ERROR_DATA = [("npz", [0, 5, 7, 8], [0, 1, 2]), ("zarr", [2, 6, 7, 9], [0, 1, 2])]


@pytest.mark.parametrize("storage_typ,not_use,labels", ClASS_CHOICE_ERROR_DATA)
def test_class_choice_error(data_dir: str, storage_typ, not_use: list, labels: list):
    data_storage = get_data_storage(storage_typ=storage_typ)
    choice_names = ChoiceNames(data_storage=data_storage, config_cv={}, labels=labels, y_dict_name="y")
    paths, not_use_path = get_paths(data_dir=data_dir, storage_typ=storage_typ, not_use=not_use)
    not_use_name = [data_storage.get_name(path=p) for p in not_use_path]
    with pytest.raises(ValueError, match="Check your data. Can't find enough files with all labels inside!"):
        choice_names.class_choice(paths=paths, paths_names=[data_storage.get_name(path=p) for p in paths],
                                  excepts=not_use_name)
