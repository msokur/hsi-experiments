import pytest
import os
import numpy as np

from data_utils.data_archive import DataArchiveZARR, DataArchiveNPZ
from data_utils.scaler import StandardScalerTransposed, NormalizerScaler
from testspace.data_utils.conftest import (
    DATA_1D_X_0, DATA_1D_X_1, DATA_3D_X_0, DATA_y_0, DATA_y_1, DATA_i
)

SCALER_FILE = "test.scaler"

X_1D_CONCAT = np.concatenate((DATA_1D_X_0, DATA_1D_X_1), axis=0)
Y_CONCAT = np.concatenate((DATA_y_0, DATA_y_1), axis=0)
I_CONCAT = np.concatenate((DATA_i, DATA_i), axis=0)

MEAN = np.mean(a=DATA_1D_X_0, axis=1)
STD = np.std(a=DATA_1D_X_0, axis=1)
L2 = np.sqrt(np.sum(DATA_1D_X_0 * DATA_1D_X_0, axis=1))
SNV_1D_SCALED = np.full(DATA_1D_X_0.shape, fill_value=np.nan)
L2_NORM_1D_SCALD = np.full(DATA_1D_X_0.shape, fill_value=np.nan)
for ax in range(DATA_1D_X_0.shape[0]):
    SNV_1D_SCALED[ax] = (DATA_1D_X_0[ax] - MEAN[ax]) / STD[ax]
    L2_NORM_1D_SCALD[ax] = DATA_1D_X_0[ax] / L2[ax]

SNV_3D_SCALED = np.array([list(x) * 9 for x in SNV_1D_SCALED]).reshape((SNV_1D_SCALED.shape[0], 3, 3,
                                                                        SNV_1D_SCALED.shape[-1]))
L2_NORM_3D_SCALD = np.array([list(x) * 9 for x in L2_NORM_1D_SCALD]).reshape((L2_NORM_1D_SCALD.shape[0], 3, 3,
                                                                              L2_NORM_1D_SCALD.shape[-1]))


@pytest.fixture
def delete_scaler_file(data_dir: str):
    yield
    delete_file = os.path.join(data_dir, SCALER_FILE)
    if os.path.exists(delete_file):
        os.remove(os.path.join(data_dir, SCALER_FILE))


def get_data_archive(typ: str):
    if typ == "zarr":
        return DataArchiveZARR()
    else:
        return DataArchiveNPZ()


X_Y_CONCATENATE_DATA = [(NormalizerScaler, "npz", "1d"), (NormalizerScaler, "npz", "3d"),
                        (NormalizerScaler, "zarr", "1d"), (NormalizerScaler, "zarr", "3d"),
                        (StandardScalerTransposed, "npz", "1d"), (StandardScalerTransposed, "npz", "3d"),
                        (StandardScalerTransposed, "zarr", "1d"), (StandardScalerTransposed, "zarr", "3d")]


@pytest.mark.parametrize("scaler_class,data_typ,data_shape", X_Y_CONCATENATE_DATA)
def test_X_y_concatenate(delete_scaler_file, data_dir: str, scaler_class, data_typ: str, data_shape: str):
    scaler = scaler_class(preprocessed_path=data_dir, data_archive=get_data_archive(typ=data_typ),
                          scaler_file=SCALER_FILE)
    scaler.preprocessed_path = os.path.join(data_dir, data_typ + "_file", data_shape)

    values = scaler.X_y_concatenate()

    for val, res in zip(values, (X_1D_CONCAT, Y_CONCAT, I_CONCAT)):
        assert (val == res).all()


SCALE_X_DATA = [(NormalizerScaler, DATA_1D_X_0, L2_NORM_1D_SCALD), (NormalizerScaler, DATA_3D_X_0, L2_NORM_3D_SCALD),
                (StandardScalerTransposed, DATA_1D_X_0, SNV_1D_SCALED),
                (StandardScalerTransposed, DATA_3D_X_0, SNV_3D_SCALED)]


@pytest.mark.parametrize("scaler_class,x_raw,x_result", SCALE_X_DATA)
def test_scale_X(delete_scaler_file, data_dir: str, scaler_class, x_raw: np.ndarray, x_result: np.ndarray):
    scaler = scaler_class(preprocessed_path=data_dir, data_archive=DataArchiveNPZ, scaler_file=SCALER_FILE)
    X = scaler.scale_X(X=x_raw)
    assert (x_result == X).all()
