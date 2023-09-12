import pytest
import os
import zarr
from shutil import rmtree
import numpy as np

from data_utils.data_loaders.data_loader import DataLoader
from data_utils.data_archive.data_archive_zarr import DataArchiveZARR

ARR_DATA_1 = {"X": np.array([[0, 1], [2, 3], [4, 5]]), "y": np.array([0, 1, 2]),
              "indexes_in_datacube": np.array([(0, 0), (1, 1), (2, 2)])}

ARR_DATA_2 = {"X": np.array([[5, 6], [7, 8], [9, 10]]), "y": np.array([2, 1, 0]),
              "indexes_in_datacube": np.array([(1, 0), (1, 2), (0, 2)])}

ARR_DATA_3 = {"X": np.array([[[[0, 0], [0, 1], [0, 2]], [[0, 3], [0, 4], [0, 5]], [[0, 6], [0, 7], [0, 8]]],
                             [[[1, 0], [1, 1], [1, 2]], [[1, 3], [1, 4], [1, 5]], [[1, 6], [1, 7], [1, 8]]],
                             [[[2, 0], [2, 1], [2, 2]], [[2, 3], [2, 4], [2, 5]], [[2, 6], [2, 7], [2, 8]]]]),
              "y": np.array([0, 1, 2]), "indexes_in_datacube": np.array([(1, 0), (1, 2), (0, 2)])}


@pytest.fixture(scope="function", autouse=True)
def delete_archive(zarr_test_dir: str):
    yield
    if os.path.exists(zarr_test_dir):
        rmtree(zarr_test_dir)


@pytest.fixture
def zarr_test_dir(zarr_data_dir: str) -> str:
    return os.path.join(zarr_data_dir, "test")


@pytest.fixture
def dataloader() -> DataLoader:
    data_archive = DataArchiveZARR()
    return DataLoader(data_archive=data_archive, config_dataloader={"FILE_EXTENSION": ".dat"}, config_paths={})


def test_get_name(zarr_test_dir: str, dataloader):
    path = zarr_test_dir + "/pat_1"
    assert dataloader.get_name(path=path) == "pat_1"


GET_PATHS_DATA = [([ARR_DATA_1, ARR_DATA_2, ARR_DATA_3], ["pat_1", "pat_2", "pat_3"])]


@pytest.mark.parametrize("values,names", GET_PATHS_DATA)
def test_get_paths(zarr_test_dir: str, dataloader, values: list, names: list):
    result = []
    for value, name in zip(values, names):
        dataloader.X_y_dict_save_to_archive(destination_path=zarr_test_dir, values=value, name=name)
        result.append(f"{zarr_test_dir}/{name}")

    elem = dataloader.get_paths(root_path=zarr_test_dir)
    assert elem == result


X_Y_DICT_SAVE_DATA_TREE = [([ARR_DATA_1, ARR_DATA_2], ["pat_one", "pat_two"],
                            ["pat_one", "X", "indexes_in_datacube", "y",  "pat_two", "X", "indexes_in_datacube",  "y"]),
                           ([ARR_DATA_1], ["pat_one"], ["pat_one", "X", "indexes_in_datacube", "y"]),
                           ([ARR_DATA_3], ["pat_three"], ["pat_three", "X", "indexes_in_datacube", "y"])]


@pytest.mark.parametrize("values,names,result", X_Y_DICT_SAVE_DATA_TREE)
def test_X_y_dict_save_to_archive_tree(zarr_test_dir: str, dataloader, values: list, names: list,
                                       result: list):
    for value, name in zip(values, names):
        dataloader.X_y_dict_save_to_archive(destination_path=zarr_test_dir, values=value, name=name)

    z_archive = zarr.open_group(store=zarr_test_dir, mode="r")
    elem = []
    for group in z_archive.group_keys():
        elem.append(group)
        for sub_key in z_archive[group].array_keys():
            elem.append(sub_key)

    assert elem == result


X_Y_DICT_SAVE_DATA_CHUNKS = [(ARR_DATA_1, "pat_one", [(1000, 2), (1000, 2), (1000,)]),
                             (ARR_DATA_3, "pat_three", [(1000, 1, 1, 2), (1000, 2), (1000,)])]


@pytest.mark.parametrize("values,name,result", X_Y_DICT_SAVE_DATA_CHUNKS)
def test_X_y_dict_save_to_archive_chunks(zarr_test_dir: str, dataloader, values: dict[str, np.ndarray],
                                         name: str, result: list):
    dataloader.X_y_dict_save_to_archive(destination_path=zarr_test_dir, values=values, name=name)

    z_archive = zarr.open_group(store=zarr_test_dir, mode="r")
    elem = []
    for group in z_archive.group_keys():
        for sub_key in z_archive[group].array_keys():
            elem.append(z_archive[group][sub_key].info.obj.chunks)

    assert elem == result
