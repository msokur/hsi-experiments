import numpy as np
import zarr
import pytest
import os
import pickle

from data_utils.weights import Weights
from data_utils.data_archive import DataArchiveNPZ, DataArchiveZARR


WEIGHTS_RESULT = np.array([[8, 8, 8, 8], [4, 4, 0, 0]])

FILENAME = "test.weights"

LABELS = [0, 1, 2, 3]

Y_DICT_NAME = "y"

WEIGHT_DICT_NAME = "weights"

WEIGHTS_SAVE_RESULT = {"weights": WEIGHTS_RESULT,
                       "sum": 24,
                       "quantities": np.array([[3, 3, 3, 3], [6, 6, 0, 0]])}

WEIGHT_DATA_SAVE_RESULT = np.array([np.full(shape=12, fill_value=8), np.full(shape=12, fill_value=4)])


def get_data_archive(typ: str, path: str):
    if typ == "npz":
        return DataArchiveNPZ(archive_path=path)
    elif typ == "zarr":
        return DataArchiveZARR(archive_path=path, archive_name="patients_data.zarr")


def test_weights_get_from_file(data_dir: str):
    weights = Weights(filename="weights_test.weights", data_archive=get_data_archive(typ="npz", path=data_dir),
                      labels=LABELS).weights_get_from_file(root_path=data_dir)
    assert (weights == WEIGHTS_RESULT).all()


def test_weights_get_from_file_error(data_dir: str):
    with pytest.raises(ValueError, match=f"No .weights file was found in the directory, check given path!"):
        Weights(filename=FILENAME, data_archive=get_data_archive(typ="npz", path=data_dir),
                labels=LABELS).weights_get_from_file(root_path=data_dir)


WEIGHT_GET_OR_SAVE_DATA = [("npz", "1d"),
                           ("zarr", "1d")]


@pytest.mark.parametrize("typ,folder", WEIGHT_GET_OR_SAVE_DATA)
def test_weights_get_or_save_only_get(data_dir: str, typ: str, folder: str):
    main_path = os.path.join(data_dir, f"{typ}_file", folder)
    weights_class = Weights(filename=FILENAME, data_archive=get_data_archive(typ=typ, path=main_path), labels=LABELS)
    weights = weights_class.weights_get_or_save(root_path=main_path)
    os.remove(os.path.join(main_path, FILENAME))
    assert (weights == WEIGHTS_RESULT).all()


@pytest.mark.parametrize("typ,folder", WEIGHT_GET_OR_SAVE_DATA)
def test_weights_get_or_save_only_save(data_dir: str, typ: str, folder: str):
    main_path = os.path.join(data_dir, f"{typ}_file", folder)
    weights_class = Weights(filename=FILENAME, data_archive=get_data_archive(typ=typ, path=main_path), labels=LABELS)
    weights_class.weights_get_or_save(root_path=main_path)
    data = pickle.load(open(os.path.join(main_path, FILENAME), 'rb'))
    os.remove(os.path.join(main_path, FILENAME))
    for (k0, v0), (k1, v1) in zip(data.items(), WEIGHTS_SAVE_RESULT.items()):
        assert k0 == k1 and (v0 == v1).all()


def test_weighted_data_save_npz(npz_1d_data_dir: str):
    weight = Weights(filename=FILENAME, data_archive=get_data_archive(typ="npz", path=npz_1d_data_dir), labels=LABELS,
                     y_dict_name=Y_DICT_NAME, weight_dict_name=WEIGHT_DICT_NAME)
    weight.weighted_data_save(root_path=npz_1d_data_dir, weights=WEIGHTS_RESULT)
    data_weights = []
    for file in ["data_test_0.npz", "data_test_1.npz"]:
        data = np.load(os.path.join(npz_1d_data_dir, file))
        data_weights.append(data[WEIGHT_DICT_NAME])
        np.savez(file=os.path.join(npz_1d_data_dir, file), **{k: v for k, v in data.items() if k != WEIGHT_DICT_NAME})
    assert (data_weights == WEIGHT_DATA_SAVE_RESULT).all()


def test_weighted_data_save_zarr(zarr_1d_data_dir: str):
    weight = Weights(filename=FILENAME, data_archive=get_data_archive(typ="zarr", path=zarr_1d_data_dir), labels=LABELS,
                     y_dict_name=Y_DICT_NAME, weight_dict_name=WEIGHT_DICT_NAME)
    weight.weighted_data_save(root_path=zarr_1d_data_dir, weights=WEIGHTS_RESULT)
    load_zarr = os.path.join(zarr_1d_data_dir, "patients_data.zarr")
    data_weights = []
    for file in ["data_test_0", "data_test_1"]:
        data = zarr.open_group(store=os.path.join(load_zarr, file), mode="a")
        data_weights.append(data[WEIGHT_DICT_NAME][...].copy())
        data.pop(key=WEIGHT_DICT_NAME)

    assert (data_weights == WEIGHT_DATA_SAVE_RESULT).all()
