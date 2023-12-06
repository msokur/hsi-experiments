import pytest
import json
import os
import numpy as np

from configuration.parameter import (
    TFR_META_EXTENSION, FEATURE_X_AXIS_1, FEATURE_X_AXIS_2, FEATURE_X, FEATURE_Y, FEATURE_IDX_CUBE, FEATURE_WEIGHTS,
    FEATURE_SAMPLES, FEATURE_SPEC, FEATURE_PAT_NAME, PAT_NAME_SEPERATOR, PAT_NAME_ENCODING, FEATURE_PAT_IDX
)
from data_utils.tfrecord import write_meta_info

NAME_0 = "test_0"
NAME_IDX_0 = 0
NAME_1 = "test_1"
NAME_IDX_1 = 1
NAME_2 = "test_2"
NAME_IDX_2 = 2
TOTAL_SAMPLES = 100
X_SHAPE_1D = (TOTAL_SAMPLES, 10)
X_SHAPE_3D = (TOTAL_SAMPLES, 3, 3, 10)

LABEL_LIST = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1], dtype=np.int64)
NAMES_LIST = np.array([NAME_0, NAME_1, NAME_2, NAME_2, NAME_1, NAME_0, NAME_1, NAME_2, NAME_0])
NAME_IDX_LIST = np.array([NAME_IDX_0, NAME_IDX_1, NAME_IDX_2, NAME_IDX_2, NAME_IDX_1, NAME_IDX_0, NAME_IDX_1,
                          NAME_IDX_2, NAME_IDX_0], dtype=np.int64)

FILE_NAME = "test_meta"


@pytest.mark.parametrize("X_shape", [X_SHAPE_1D, X_SHAPE_3D])
def test_write_meta_info(delete_meta_info, data_dir: str, X_shape: tuple):
    write_meta_info(save_dir=data_dir, file_name=FILE_NAME, labels=LABEL_LIST, names=NAMES_LIST,
                    names_idx=NAME_IDX_LIST, X_shape=X_shape)
    meta_info = json.load(open(os.path.join(data_dir, FILE_NAME + TFR_META_EXTENSION)))
    result = _get_meta_info(d3=True if X_shape.__len__() > 2 else False)
    assert meta_info == result


def test_write_meta_info_error():
    ERROR_NAME_LIST = NAMES_LIST.copy()
    ERROR_NAME_LIST[1] = NAME_0
    with pytest.raises(ValueError, match=f"Too many indexes for patient name '{NAME_0}'!"):
        write_meta_info(save_dir="", file_name=FILE_NAME, labels=LABEL_LIST, names=ERROR_NAME_LIST,
                        names_idx=NAME_IDX_LIST, X_shape=X_SHAPE_1D)
@pytest.fixture
def delete_meta_info(data_dir: str):
    yield
    os.remove(os.path.join(data_dir, FILE_NAME + TFR_META_EXTENSION))


def _get_meta_info(d3: bool) -> dict:
    if d3:
        meta_info = _base_meta_data(X_shape=list(X_SHAPE_3D)[1:])
        meta_info[f"{FEATURE_X_AXIS_1}_size_dtype"] = "int64"
        meta_info[f"{FEATURE_X_AXIS_2}_size_dtype"] = "int64"
    else:
        meta_info = _base_meta_data(X_shape=list(X_SHAPE_1D)[1:])

    return meta_info


def _base_meta_data(X_shape: list) -> dict:
    return {
        "total_samples": TOTAL_SAMPLES,
        f"{FEATURE_X}_shape": X_shape,
        f"{FEATURE_X}_dtype": "float32",
        f"{FEATURE_Y}_dtype": "int64",
        f"{FEATURE_IDX_CUBE}_dtype": "int64",
        f"{FEATURE_WEIGHTS}_dtype": "float32",
        f"{FEATURE_SAMPLES}_dtype": "int64",
        f"{FEATURE_SPEC}_size_dtype": "int64",
        f"{FEATURE_PAT_NAME}_seperator": PAT_NAME_SEPERATOR,
        f"{FEATURE_PAT_NAME}_encoding": PAT_NAME_ENCODING,
        f"{FEATURE_PAT_IDX}_dtype": "int64",
        "samples_per_patient_name": {
            NAME_0: {
                FEATURE_PAT_IDX: NAME_IDX_0,
                "0": 1,
                "1": 2,
            },
            NAME_1: {
                FEATURE_PAT_IDX: NAME_IDX_1,
                "0": 1,
                "1": 1,
                "2": 1,
            },
            NAME_2: {
                FEATURE_PAT_IDX: NAME_IDX_2,
                "0": 1,
                "2": 1,
                "3": 1
            }
        }
    }
