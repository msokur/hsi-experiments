import numpy as np
import pytest
import zarr
import os
from shutil import rmtree

from testspace.data_utils.conftest import (
    DATA_1D_X_0, DATA_1D_X_1, DATA_3D_X_0, DATA_3D_X_1, DATA_y_0, DATA_y_1, DATA_i, DATA_1D_0, DATA_1D_1, DATA_3D_0,
    DATA_3D_1
)
from data_utils.data_archive import DataArchiveZARR, DataArchiveNPZ, DataArchive


GET_PATH_TEST_DATA = [(DataArchiveNPZ(), "npz", "data_test_0.npz"),
                      (DataArchiveZARR(), "zarr", "data_test_0")]


def get_file(typ: str, path: str):
    if typ == "npz":
        return np.load(file=path)
    elif typ == "zarr":
        return zarr.open_group(store=path, mode="r")


@pytest.mark.parametrize("data_archive,typ,data_name", GET_PATH_TEST_DATA)
def test_get_path(data_dir: str, data_archive: DataArchive, typ: str, data_name: str):
    file_path = os.path.join(data_dir, f"{typ}_file", "1d", data_name)
    file = get_file(typ=typ, path=file_path)
    assert data_archive.get_path(file=file) == file_path


GET_PATHS_TEST_DATA = [(DataArchiveNPZ(), "npz", "1d", ["data_test_0.npz", "data_test_1.npz"]),
                       (DataArchiveNPZ(), "npz", "3d", ["data_test_0.npz", "data_test_1.npz"]),
                       (DataArchiveZARR(), "zarr", "1d", ["data_test_0", "data_test_1"]),
                       (DataArchiveZARR(), "zarr", "3d", ["data_test_0", "data_test_1"])]


@pytest.mark.parametrize("data_archive,typ,patch,data_names", GET_PATHS_TEST_DATA)
def test_get_paths(data_dir: str, data_archive: DataArchive, typ: str, patch: str, data_names: list):
    result_paths = []
    root_path = os.path.join(data_dir, f"{typ}_file", patch)
    for name in data_names:
        result_paths.append(os.path.join(root_path, name))
    assert data_archive.get_paths(archive_path=root_path) == result_paths


GET_NAME_TEST_DATA = [(DataArchiveNPZ(), r"C:\folder0\folder1\data.npz", "data"),
                      (DataArchiveNPZ(), "/folder0/folder1/data.npz", "data"),
                      (DataArchiveZARR(), r"C:\folder0\folder1\data", "data"),
                      (DataArchiveZARR(), "/folder0/folder1/data", "data"),
                      (DataArchiveZARR(), "/folder0/folder1/data.png.npz", "data.png"),
                      (DataArchiveZARR(), "/folder0.1/folder1/data", "data"),
                      (DataArchiveNPZ(), r"C:\folder0\folder1\data.png.npz", "data.png"),
                      (DataArchiveNPZ(), r"C:\folder0.1\folder1\data", "data")]


@pytest.mark.parametrize("data_archive,path,result", GET_NAME_TEST_DATA)
def test_get_name(data_archive: DataArchive, path: str, result: str):
    assert data_archive.get_name(path=path) == result


ALL_DATA_GENERATOR_TEST_DATA = [(DataArchiveNPZ(), "npz", "1d", [DATA_1D_0, DATA_1D_1]),
                                (DataArchiveNPZ(), "npz", "3d", [DATA_3D_0, DATA_3D_1]),
                                (DataArchiveZARR(), "zarr", "1d", [DATA_1D_0, DATA_1D_1]),
                                (DataArchiveZARR(), "zarr", "3d", [DATA_3D_0, DATA_3D_1])]


@pytest.mark.parametrize("data_archive,typ,patch,results", ALL_DATA_GENERATOR_TEST_DATA)
def test_all_data_generator_keys(data_dir: str, data_archive: DataArchive, typ: str, patch: str, results: list):
    data_generator = data_archive.all_data_generator(archive_path=os.path.join(data_dir, f"{typ}_file", patch))
    for data_gen, result in zip(data_generator, results):
        assert data_gen.keys() == result.keys()


@pytest.mark.parametrize("data_archive,typ,patch,results", ALL_DATA_GENERATOR_TEST_DATA)
def test_all_data_generator_values(data_dir: str, data_archive: DataArchive, typ: str, patch: str, results: list):
    data_generator = data_archive.all_data_generator(archive_path=os.path.join(data_dir, f"{typ}_file", patch))
    for data_gen, result in zip(data_generator, results):
        for k in data_gen.keys():
            assert (data_gen[k][...] == result[k]).all()


GET_DATAS_TEST_DATA = [(DataArchiveNPZ(), "npz", "1d", "data_test_0.npz", DATA_1D_0),
                       (DataArchiveNPZ(), "npz", "3d", "data_test_1.npz", DATA_3D_1),
                       (DataArchiveZARR(), "zarr", "1d", "data_test_1", DATA_1D_1),
                       (DataArchiveZARR(), "zarr", "3d", "data_test_0", DATA_3D_0)]


@pytest.mark.parametrize("data_archive,typ,patch,group_name,result", GET_DATAS_TEST_DATA)
def test_get_datas_keys(data_dir: str, data_archive: DataArchive, typ: str, patch: str, group_name: str, result: dict):
    file_path = os.path.join(data_dir, f"{typ}_file", patch, group_name)
    datas = data_archive.get_datas(data_path=file_path)
    assert datas.keys() == result.keys()


@pytest.mark.parametrize("data_archive,typ,patch,group_name,result", GET_DATAS_TEST_DATA)
def test_get_datas_values(data_dir: str, data_archive: DataArchive, typ: str, patch: str, group_name: str,
                          result: dict):
    file_path = os.path.join(data_dir, f"{typ}_file", patch, group_name)
    datas = data_archive.get_datas(data_path=file_path)
    for k in datas.keys():
        assert (datas[k][...] == result[k]).all()


GET_DATA_TEST_DATA = [(DataArchiveNPZ(), "npz", "1d", "data_test_0.npz", "X", DATA_1D_X_0),
                      (DataArchiveNPZ(), "npz", "3d", "data_test_1.npz", "X", DATA_3D_X_1),
                      (DataArchiveNPZ(), "npz", "1d", "data_test_0.npz", "y", DATA_y_0),
                      (DataArchiveNPZ(), "npz", "1d", "data_test_0.npz", "indexes_in_datacube", DATA_i),
                      (DataArchiveZARR(), "zarr", "1d", "data_test_1", "X", DATA_1D_X_1),
                      (DataArchiveZARR(), "zarr", "3d", "data_test_0", "X", DATA_3D_X_0),
                      (DataArchiveZARR(), "zarr", "3d", "data_test_1", "y", DATA_y_1),
                      (DataArchiveZARR(), "zarr", "3d", "data_test_1", "indexes_in_datacube", DATA_i)]


@pytest.mark.parametrize("data_archive,typ,patch,data_name,array_name,result", GET_DATA_TEST_DATA)
def test_get_data(data_dir: str, data_archive: DataArchive, typ: str, patch: str, data_name: str, array_name: str,
                  result: list):
    file_path = os.path.join(data_dir, f"{typ}_file", patch, data_name)
    assert (data_archive.get_data(data_path=file_path, data_name=array_name)[...] == result).all()


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


SAVE_GROUP_TEST_DATA = [(DataArchiveNPZ(), ".npz"),
                        (DataArchiveZARR(), ".zarr")]


@pytest.mark.parametrize("data_archive,ext", SAVE_GROUP_TEST_DATA)
def test_save_group_exist(delete_save_group_archive, save_group_path: str, group_name: str, data_archive: DataArchive,
                          ext: str):
    data_archive.save_group(save_path=save_group_path, group_name=group_name + ext, datas=DATA_1D_0)
    assert os.path.exists(os.path.join(save_group_path, group_name + ext))


@pytest.mark.parametrize("data_archive,ext", SAVE_GROUP_TEST_DATA)
def test_save_group_data(delete_save_group_archive, save_group_path: str, group_name: str, data_archive: DataArchive,
                         ext: str):
    data_archive.save_group(save_path=save_group_path, group_name=group_name + ext, datas=DATA_1D_0)
    datas = data_archive.get_datas(os.path.join(save_group_path, group_name + ext))
    for (result_key, result_value), (k, v) in zip(datas.items(), DATA_1D_0.items()):
        assert result_key == k and (result_value[...] == v).all()


@pytest.mark.parametrize("data_archive,ext", SAVE_GROUP_TEST_DATA)
def test_save_data(delete_save_group_archive, save_group_path: str, group_name: str, data_archive: DataArchive,
                   ext: str):
    result = DATA_1D_0.copy()
    result["test"] = DATA_y_0
    data_archive.save_group(save_path=save_group_path, group_name=group_name + ext, datas=DATA_1D_0)
    data_archive.save_data(save_path=os.path.join(save_group_path, group_name + ext), data_name="test", data=DATA_y_0)
    for k, v in data_archive.get_datas(data_path=os.path.join(save_group_path, group_name + ext)).items():
        assert (v[...] == result[k]).all()


SAVE_DATA_TEST_DATA = [(DataArchiveNPZ(), ".npz", FileNotFoundError),
                       (DataArchiveZARR(), ".zarr", zarr.errors.GroupNotFoundError)]


@pytest.mark.parametrize("data_archive,ext,error", SAVE_DATA_TEST_DATA)
def test_save_data_error(save_group_path: str, group_name: str, data_archive: DataArchive, ext: str, error):
    with pytest.raises(error):
        data_archive.save_data(save_path=os.path.join(save_group_path, group_name + ext), data_name="test",
                               data=DATA_y_0)


@pytest.mark.parametrize("data_archive,ext", SAVE_GROUP_TEST_DATA)
def test_delete_archive(save_group_path: str, group_name: str, data_archive: DataArchive, ext: str):
    data_archive.save_group(save_path=save_group_path, group_name=group_name + ext, datas=DATA_1D_0)
    exist = os.path.exists(os.path.join(save_group_path, group_name + ext))
    data_archive.delete_archive(delete_path=save_group_path)
    deleted = not os.path.exists(os.path.join(save_group_path, group_name + ext))
    assert exist and deleted
