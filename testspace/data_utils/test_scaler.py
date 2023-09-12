import numpy as np
import pytest
import os

from data_utils.scaler import StandardScaler, SNV
from data_utils.data_archive.data_archive_zarr import DataArchiveZARR
from data_utils.data_archive.data_archive_npz import DataArchiveNPZ

X_0 = np.array(np.arange(start=0, stop=120, step=1)).reshape((12, 10))

X_1 = np.array(np.arange(start=120, stop=240, step=1)).reshape((12, 10))

X = np.concatenate((X_0, X_1), axis=0)

SAMPLES = np.int64(24)

FEATURES = 10

SHAPE_1D = (12, FEATURES)

SHAPE_3D = (12, 3, 3, FEATURES)

SNV_MEAN_RESULT = np.array([115., 116., 117., 118., 119., 120., 121., 122., 123., 124.])

SNV_VAR_RESULT = np.array([4791.666667] * FEATURES)

SNV_STD_RESULT = np.sqrt(SNV_VAR_RESULT)

X_1_SNV_RESULT = np.array([[var] * FEATURES for var in np.arange(start=0.072232,
                                                                 stop=1.805788,
                                                                 step=0.144463)]).reshape((12, 10))

X_0_SNV_RESULT = np.flip(X_1_SNV_RESULT * -1)


def get_data_archive(typ: str):
    if typ == "npz":
        return DataArchiveNPZ()
    elif typ == "zarr":
        return DataArchiveZARR()


@pytest.fixture(scope="function", autouse=True)
def delete_scaler_file(npz_data_dir: str, zarr_data_dir: str):
    yield
    for size in ["1d", "3d"]:
        if os.path.exists(os.path.join(npz_data_dir, size, "scaler.scaler")):
            os.remove(os.path.join(npz_data_dir, size, "scaler.scaler"))

        if os.path.exists(os.path.join(zarr_data_dir, size, "scaler.scaler")):
            os.remove(os.path.join(zarr_data_dir, size, "scaler.scaler"))


SNV_TEST_DATA = [(StandardScaler, "1d", "npz"), (StandardScaler, "3d", "npz"), (StandardScaler, "1d", "zarr"),
                 (StandardScaler, "3d", "zarr"),
                 (SNV, "1d", "npz"), (SNV, "3d", "npz"), (SNV, "1d", "zarr"), (SNV, "3d", "zarr")]


@pytest.mark.parametrize("scaler_class,folder,typ", SNV_TEST_DATA)
def test_standard_scaler_samples(data_dir: str, scaler_class, folder: str, typ: str):
    path = os.path.join(data_dir, f"{typ}_file", folder)
    scaler = scaler_class(preprocessed_path=path, data_archive=get_data_archive(typ=typ))
    assert scaler.scaler.n_samples_seen_ == SAMPLES


@pytest.mark.parametrize("scaler_class,folder,typ", SNV_TEST_DATA)
def test_standard_scaler_features(data_dir: str, scaler_class, folder: str, typ: str):
    path = os.path.join(data_dir, f"{typ}_file", folder)
    scaler = scaler_class(preprocessed_path=path, data_archive=get_data_archive(typ=typ))
    assert scaler.scaler.n_features_in_ == FEATURES


@pytest.mark.parametrize("scaler_class,folder,typ", SNV_TEST_DATA)
def test_standard_scaler_mean(data_dir: str, scaler_class, folder: str, typ: str):
    path = os.path.join(data_dir, f"{typ}_file", folder)
    scaler = scaler_class(preprocessed_path=path, data_archive=get_data_archive(typ=typ))
    assert (scaler.scaler.mean_ == SNV_MEAN_RESULT).all()


@pytest.mark.parametrize("scaler_class,folder,typ", SNV_TEST_DATA)
def test_standard_scaler_var(data_dir: str, scaler_class, folder: str, typ: str):
    path = os.path.join(data_dir, f"{typ}_file", folder)
    scaler = scaler_class(preprocessed_path=path, data_archive=get_data_archive(typ=typ))
    assert pytest.approx(scaler.scaler.var_, 0.000001) == SNV_VAR_RESULT


@pytest.mark.parametrize("scaler_class,folder,typ", SNV_TEST_DATA)
def test_standard_scaler_std(data_dir: str, scaler_class, folder: str, typ: str):
    path = os.path.join(data_dir, f"{typ}_file", folder)
    scaler = scaler_class(preprocessed_path=path, data_archive=get_data_archive(typ=typ))
    assert pytest.approx(scaler.scaler.scale_, 0.000001) == SNV_STD_RESULT


SNV_GET_DATA = [("npz", "1d", SHAPE_1D), ("npz", "3d", SHAPE_3D), ("zarr", "1d", SHAPE_1D), ("zarr", "3d", SHAPE_3D)]


@pytest.mark.parametrize("typ,folder,shape", SNV_GET_DATA)
def test_standard_scaler_get_samples_features_shape(data_dir: str, typ: str, folder: str, shape: tuple):
    path = os.path.join(data_dir, f"{typ}_file", folder)
    scaler = SNV(preprocessed_path=path, data_archive=get_data_archive(typ=typ))
    assert scaler.get_samples_features_shape() == (SAMPLES, FEATURES, shape)


@pytest.mark.parametrize("typ,folder,shape", SNV_GET_DATA)
def test_standard_scaler_get_mean(data_dir: str, typ: str, folder: str, shape: tuple):
    path = os.path.join(data_dir, f"{typ}_file", folder)
    scaler = SNV(preprocessed_path=path, data_archive=get_data_archive(typ=typ))
    assert (scaler.get_mean(samples=SAMPLES, features=FEATURES, shape=shape) == SNV_MEAN_RESULT).all()


@pytest.mark.parametrize("typ,folder,shape", SNV_GET_DATA)
def test_standard_scaler_get_var(data_dir: str, typ: str, folder: str, shape: tuple):
    path = os.path.join(data_dir, f"{typ}_file", folder)
    scaler = SNV(preprocessed_path=path, data_archive=get_data_archive(typ=typ))
    assert pytest.approx(scaler.get_var(mean=SNV_MEAN_RESULT, samples=SAMPLES, features=FEATURES,
                                        shape=shape), 0.000001) == SNV_VAR_RESULT


@pytest.mark.parametrize("typ,folder,shape", SNV_GET_DATA)
def test_standard_scaler_get_std(data_dir: str, typ: str, folder: str, shape: tuple):
    path = os.path.join(data_dir, f"{typ}_file", folder)
    scaler = SNV(preprocessed_path=path, data_archive=get_data_archive(typ=typ))
    assert pytest.approx(scaler.get_std(var=SNV_VAR_RESULT), 0.000001) == SNV_STD_RESULT
