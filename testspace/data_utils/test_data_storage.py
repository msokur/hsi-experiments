import numpy as np
import pytest
import zarr
import os
from shutil import rmtree

from testspace.data_utils.conftest import (
    DATA_1D_X_0, DATA_1D_X_1, DATA_3D_X_0, DATA_3D_X_1, DATA_y_0, DATA_y_1, DATA_i, DATA_1D_0, DATA_1D_1, DATA_3D_0,
    DATA_3D_1
)
from data_utils.data_storage import DataStorageZARR, DataStorageNPZ, DataStorage


GET_PATH_TEST_DATA = [(DataStorageNPZ(), "npz", "data_test_0.npz"),
                      (DataStorageZARR(), "zarr", "data_test_0")]


def get_file(typ: str, path: str):
    if typ == "npz":
        return np.load(file=path)
    elif typ == "zarr":
        return zarr.open_group(store=path, mode="r")


@pytest.mark.parametrize("data_storage,typ,data_name", GET_PATH_TEST_DATA)
def test_get_path(data_dir: str, data_storage, typ: str, data_name: str):
    file_path = os.path.join(data_dir, f"{typ}_file", "1d", "patient", data_name)
    file = get_file(typ=typ, path=file_path)
    assert data_storage.get_path(file=file) == file_path


GET_PATHS_TEST_DATA = [(DataStorageNPZ(), "npz", "1d", ["data_test_0.npz", "data_test_1.npz"]),
                       (DataStorageNPZ(), "npz", "3d", ["data_test_0.npz", "data_test_1.npz"]),
                       (DataStorageZARR(), "zarr", "1d", ["data_test_0", "data_test_1"]),
                       (DataStorageZARR(), "zarr", "3d", ["data_test_0", "data_test_1"])]


@pytest.mark.parametrize("data_storage,typ,patch,data_names", GET_PATHS_TEST_DATA)
def test_get_paths(data_dir: str, data_storage, typ: str, patch: str, data_names: list):
    result_paths = []
    root_path = os.path.join(data_dir, f"{typ}_file", patch, "patient")
    for name in data_names:
        result_paths.append(os.path.join(root_path, name))
    print(data_storage.get_paths(storage_path=root_path))
    print(root_path)
    assert data_storage.get_paths(storage_path=root_path) == result_paths


GET_NAME_TEST_DATA = [(DataStorageNPZ(), r"C:\folder0\folder1\data.npz", "data"),
                      (DataStorageNPZ(), "/folder0/folder1/data.npz", "data"),
                      (DataStorageZARR(), r"C:\folder0\folder1\data", "data"),
                      (DataStorageZARR(), "/folder0/folder1/data", "data"),
                      (DataStorageZARR(), "/folder0/folder1/data.png.npz", "data.png"),
                      (DataStorageZARR(), "/folder0.1/folder1/data", "data"),
                      (DataStorageNPZ(), r"C:\folder0\folder1\data.png.npz", "data.png"),
                      (DataStorageNPZ(), r"C:\folder0.1\folder1\data", "data")]


@pytest.mark.parametrize("data_storage,path,result", GET_NAME_TEST_DATA)
def test_get_name(data_storage, path: str, result: str):
    assert data_storage.get_name(path=path) == result


ALL_DATA_GENERATOR_TEST_DATA = [(DataStorageNPZ(), "npz", "1d", [DATA_1D_0, DATA_1D_1]),
                                (DataStorageNPZ(), "npz", "3d", [DATA_3D_0, DATA_3D_1]),
                                (DataStorageZARR(), "zarr", "1d", [DATA_1D_0, DATA_1D_1]),
                                (DataStorageZARR(), "zarr", "3d", [DATA_3D_0, DATA_3D_1])]


@pytest.mark.parametrize("data_storage,typ,patch,results", ALL_DATA_GENERATOR_TEST_DATA)
def test_all_data_generator_keys(data_dir: str, data_storage, typ: str, patch: str, results: list):
    storage_path = os.path.join(data_dir, f"{typ}_file", patch, "patient")
    data_generator = data_storage.all_data_generator(storage_path=storage_path)
    for data_gen, result in zip(data_generator, results):
        assert data_gen.keys() == result.keys()


@pytest.mark.parametrize("data_storage,typ,patch,results", ALL_DATA_GENERATOR_TEST_DATA)
def test_all_data_generator_values(data_dir: str, data_storage, typ: str, patch: str, results: list):
    storage_path = os.path.join(data_dir, f"{typ}_file", patch, "patient")
    data_generator = data_storage.all_data_generator(storage_path=storage_path)
    for data_gen, result in zip(data_generator, results):
        for k in data_gen.keys():
            assert (data_gen[k][...] == result[k]).all()


GET_DATAS_TEST_DATA = [(DataStorageNPZ(), "npz", "1d", "data_test_0.npz", DATA_1D_0),
                       (DataStorageNPZ(), "npz", "3d", "data_test_1.npz", DATA_3D_1),
                       (DataStorageZARR(), "zarr", "1d", "data_test_1", DATA_1D_1),
                       (DataStorageZARR(), "zarr", "3d", "data_test_0", DATA_3D_0)]


@pytest.mark.parametrize("data_storage,typ,patch,group_name,result", GET_DATAS_TEST_DATA)
def test_get_datas_keys(data_dir: str, data_storage, typ: str, patch: str, group_name: str, result: dict):
    file_path = os.path.join(data_dir, f"{typ}_file", patch, "patient", group_name)
    datas = data_storage.get_datas(data_path=file_path)
    print(datas)
    assert datas.keys() == result.keys()


@pytest.mark.parametrize("data_storage,typ,patch,group_name,result", GET_DATAS_TEST_DATA)
def test_get_datas_values(data_dir: str, data_storage, typ: str, patch: str, group_name: str,
                          result: dict):
    file_path = os.path.join(data_dir, f"{typ}_file", patch, "patient", group_name)
    datas = data_storage.get_datas(data_path=file_path)
    for k in datas.keys():
        assert (datas[k][...] == result[k]).all()


GET_DATA_TEST_DATA = [(DataStorageNPZ(), "npz", "1d", "data_test_0.npz", "X", DATA_1D_X_0),
                      (DataStorageNPZ(), "npz", "3d", "data_test_1.npz", "X", DATA_3D_X_1),
                      (DataStorageNPZ(), "npz", "1d", "data_test_0.npz", "y", DATA_y_0),
                      (DataStorageNPZ(), "npz", "1d", "data_test_0.npz", "indexes_in_datacube", DATA_i),
                      (DataStorageZARR(), "zarr", "1d", "data_test_1", "X", DATA_1D_X_1),
                      (DataStorageZARR(), "zarr", "3d", "data_test_0", "X", DATA_3D_X_0),
                      (DataStorageZARR(), "zarr", "3d", "data_test_1", "y", DATA_y_1),
                      (DataStorageZARR(), "zarr", "3d", "data_test_1", "indexes_in_datacube", DATA_i)]


@pytest.mark.parametrize("data_storage,typ,patch,data_name,array_name,result", GET_DATA_TEST_DATA)
def test_get_data(data_dir: str, data_storage, typ: str, patch: str, data_name: str, array_name: str,
                  result: list):
    file_path = os.path.join(data_dir, f"{typ}_file", patch, "patient", data_name)
    assert (data_storage.get_data(data_path=file_path, data_name=array_name)[...] == result).all()


@pytest.fixture
def delete_save_group_archive(save_group_path: str):
    yield
    rmtree(path=save_group_path)


@pytest.fixture
def save_group_path(data_dir: str) -> str:
    return os.path.join(data_dir, "save_group")


@pytest.fixture
def group_name() -> str:
    return "save_group_testing"


SAVE_GROUP_TEST_DATA = [(DataStorageNPZ(), ".npz"),
                        (DataStorageZARR(), ".zarr")]


@pytest.mark.parametrize("data_storage,ext", SAVE_GROUP_TEST_DATA)
def test_save_group_exist(delete_save_group_archive, save_group_path: str, group_name: str, data_storage: DataStorage,
                          ext: str):
    data_storage.save_group(save_path=save_group_path, group_name=group_name + ext, datas=DATA_1D_0)
    assert os.path.exists(os.path.join(save_group_path, group_name + ext))


@pytest.mark.parametrize("data_storage,ext", SAVE_GROUP_TEST_DATA)
def test_save_group_data(delete_save_group_archive, save_group_path: str, group_name: str, data_storage: DataStorage,
                         ext: str):
    data_storage.save_group(save_path=save_group_path, group_name=group_name + ext, datas=DATA_1D_0)
    datas = data_storage.get_datas(os.path.join(save_group_path, group_name + ext))
    for (result_key, result_value), (k, v) in zip(datas.items(), DATA_1D_0.items()):
        assert result_key == k and (result_value[...] == v).all()


@pytest.mark.parametrize("data_storage,ext", SAVE_GROUP_TEST_DATA)
def test_save_data(delete_save_group_archive, save_group_path: str, group_name: str, data_storage: DataStorage,
                   ext: str):
    result = DATA_1D_0.copy()
    result["test"] = DATA_y_0
    data_storage.save_group(save_path=save_group_path, group_name=group_name + ext, datas=DATA_1D_0)
    data_storage.save_data(save_path=os.path.join(save_group_path, group_name + ext), data_name="test", data=DATA_y_0)
    for k, v in data_storage.get_datas(data_path=os.path.join(save_group_path, group_name + ext)).items():
        assert (v[...] == result[k]).all()


SAVE_DATA_TEST_DATA = [(DataStorageNPZ(), ".npz", FileNotFoundError),
                       (DataStorageZARR(), ".zarr", zarr.errors.GroupNotFoundError)]


@pytest.mark.parametrize("data_storage,ext,error", SAVE_DATA_TEST_DATA)
def test_save_data_error(save_group_path: str, group_name: str, data_storage, ext: str, error):
    with pytest.raises(error):
        data_storage.save_data(save_path=os.path.join(save_group_path, group_name + ext), data_name="test",
                               data=DATA_y_0)


@pytest.mark.parametrize("data_storage,ext", SAVE_GROUP_TEST_DATA)
def test_delete_archive(save_group_path: str, group_name: str, data_storage, ext: str):
    data_storage.save_group(save_path=save_group_path, group_name=group_name + ext, datas=DATA_1D_0)
    exist = os.path.exists(os.path.join(save_group_path, group_name + ext))
    data_storage.delete_archive(delete_path=save_group_path)
    deleted = not os.path.exists(os.path.join(save_group_path, group_name + ext))
    assert exist and deleted
