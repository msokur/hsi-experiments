from typing import List, Tuple

import os

import pytest

from data_utils.dataset.generator.batch_split import NameBatchSplit

from data_utils.data_storage import (
    DataStorageZARR,
    DataStorageNPZ,
    DataStorage
)

from testspace.data_utils.dataset.conftest import (
    DICT_NAMES,
    y_NAME,
    BATCH_FOLDER,
)


def get_batch_paths(data_storage: DataStorage, batch_size: int, data_dir: str, train_names: List[str],
                    valid_names: List[str]) -> Tuple[List[str], List[str]]:
    name_split = NameBatchSplit(data_storage=data_storage,
                                batch_size=batch_size,
                                use_labels=[0, 1, 2, 3],
                                dict_names=DICT_NAMES,
                                with_sample_weights=False)

    typ = data_storage.get_extension().replace('.', '')
    sh_path = os.path.join(data_dir, f"{typ}_file", "1d", "shuffled")
    data_paths = data_storage.get_paths(storage_path=sh_path)

    os.mkdir(os.path.join(data_dir, BATCH_FOLDER))

    train_path, valid_path = name_split.split(data_paths=data_paths,
                                              batch_save_path=os.path.join(data_dir, BATCH_FOLDER),
                                              train_names=train_names,
                                              valid_names=valid_names,
                                              train_folder="train",
                                              valid_folder="valid")

    return train_path, valid_path


FILE_SIZE_TEST_DATA = [(DataStorageNPZ(), 68),
                       (DataStorageZARR(), 55)]


@pytest.mark.parametrize("data_storage,batch_size", FILE_SIZE_TEST_DATA)
def test_split_check_file_size(_delete_batches, data_dir: str, data_storage: DataStorage, batch_size: int):
    train_path, valid_path = get_batch_paths(data_storage=data_storage,
                                             batch_size=batch_size,
                                             data_dir=data_dir,
                                             train_names=["test_0", "test_1", "test_2"],
                                             valid_names=["test_3", "test_4"])
    all_paths = train_path + valid_path

    for p in all_paths:
        data = data_storage.get_datas(data_path=p)
        assert data[y_NAME].shape[0] == batch_size


BATCH_SIZE_TEST_DATA = [(DataStorageNPZ(), 60, ["test_0", "test_1", "test_2"], ["test_3", "test_4"], [3, 2]),
                        (DataStorageNPZ(), 120, ["test_0", "test_1", "test_2"], [], [1, 0]),
                        (DataStorageZARR(), 40, ["test_0", "test_1"], ["test_2", "test_3", "test_4"], [3, 4]),
                        (DataStorageZARR(), 200, ["test_0", "test_1", "test_2"], ["test_3", "test_4"], [0, 0])]


@pytest.mark.parametrize("data_storage,batch_size,train_names,valid_names,path_lengths", BATCH_SIZE_TEST_DATA)
def test_split_check_batch_size(_delete_batches, data_dir: str, data_storage: DataStorage, batch_size: int,
                                train_names: List[str], valid_names: List[str], path_lengths: List[int]):
    train_path, valid_path = get_batch_paths(data_storage=data_storage,
                                             batch_size=batch_size,
                                             data_dir=data_dir,
                                             train_names=train_names,
                                             valid_names=valid_names)

    assert [len(train_path), len(valid_path)] == path_lengths
