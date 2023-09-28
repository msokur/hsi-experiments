import numpy as np
import pytest
import zarr
import os

from data_utils.data_archive import DataArchiveZARR, DataArchiveNPZ, DataArchive

SPEC = 10
SAMPLES = 12

DATA_1D_X_0 = np.arange(start=0, stop=SAMPLES * SPEC, step=1).reshape((SAMPLES, SPEC))
DATA_1D_X_1 = np.arange(start=SAMPLES * SPEC, stop=SAMPLES * SPEC * 2, step=1).reshape((SAMPLES, SPEC))

DATA_3D_X_0 = np.array([list(x) * 9 for x in DATA_1D_X_0]).reshape((SAMPLES, 3, 3, SPEC))
DATA_3D_X_1 = np.array([list(x) * 9 for x in DATA_1D_X_1]).reshape((SAMPLES, 3, 3, SPEC))

DATA_y_0 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
DATA_y_1 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

DATA_i = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]])

DATA_1D_0 = {"X": DATA_1D_X_0, "indexes_in_datacube": DATA_i, "y": DATA_y_0}
DATA_1D_1 = {"X": DATA_1D_X_1, "indexes_in_datacube": DATA_i, "y": DATA_y_1}

DATA_3D_0 = {"X": DATA_3D_X_0, "indexes_in_datacube": DATA_i, "y": DATA_y_0}
DATA_3D_1 = {"X": DATA_3D_X_1, "indexes_in_datacube": DATA_i, "y": DATA_y_1}


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
                      (DataArchiveZARR(), "/folder0/folder1/data", "data")]


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


@pytest.mark.parametrize("data_archive,typ,patch,data_name,result", GET_DATAS_TEST_DATA)
def test_get_datas_keys(data_dir: str, data_archive: DataArchive, typ: str, patch: str, data_name: str, result: dict):
    file_path = os.path.join(data_dir, f"{typ}_file", patch, data_name)
    datas = data_archive.get_datas(data_path=file_path)
    assert datas.keys() == result.keys()


@pytest.mark.parametrize("data_archive,typ,patch,data_name,result", GET_DATAS_TEST_DATA)
def test_get_datas_values(data_dir: str, data_archive: DataArchive, typ: str, patch: str, data_name: str, result: dict):
    file_path = os.path.join(data_dir, f"{typ}_file", patch, data_name)
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


def test_save_group():
    assert False


def test_save_data():
    assert False


def test_delete_archive():
    assert False
