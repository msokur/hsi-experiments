import pytest
import json
import os
import numpy as np

from glob import glob

from configuration.parameter import (
    TFR_META_EXTENSION, GEN_META_EXTENSION,  FEATURE_X_AXIS_1, FEATURE_X_AXIS_2, TOTAL_SAMPLES, FEATURE_X, FEATURE_Y,
    FEATURE_IDX_CUBE, FEATURE_WEIGHTS, FEATURE_SAMPLES, FEATURE_SPEC, FEATURE_PAT_NAME, PAT_NAME_SEPERATOR,
    PAT_NAME_ENCODING, FEATURE_PAT_IDX, SAMPLES_PER_NAME
)
from data_utils.dataset.meta_files import write_meta_info

NAME_0 = "test_0"
NAME_IDX_0 = 0
NAME_1 = "test_1"
NAME_IDX_1 = 1
NAME_2 = "test_2"
NAME_IDX_2 = 2
TOTAL_SAMPLES_ = 100
X_SHAPE_1D = (TOTAL_SAMPLES_, 10)
X_SHAPE_3D = (TOTAL_SAMPLES_, 3, 3, 10)

LABEL_LIST = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1], dtype=np.int64)
NAMES_LIST = np.array([NAME_0, NAME_1, NAME_2, NAME_2, NAME_1, NAME_0, NAME_1, NAME_2, NAME_0])
NAME_IDX_LIST = np.array([NAME_IDX_0, NAME_IDX_1, NAME_IDX_2, NAME_IDX_2, NAME_IDX_1, NAME_IDX_0, NAME_IDX_1,
                          NAME_IDX_2, NAME_IDX_0], dtype=np.int64)

FILE_NAME = "test_meta"

WRITE_META_INFO_DATA = [(X_SHAPE_1D, "tfr"), (X_SHAPE_3D, "tfr"),
                        (X_SHAPE_1D, "generator"), (X_SHAPE_3D, "generator")]


@pytest.mark.parametrize("X_shape,typ", WRITE_META_INFO_DATA)
def test_write_meta_info(delete_meta_info, data_dir: str, X_shape: tuple, typ: str):
    write_meta_info(save_dir=data_dir, file_name=FILE_NAME, labels=LABEL_LIST, names=NAMES_LIST,
                    names_idx=NAME_IDX_LIST, X_shape=X_shape, typ=typ)
    meta_file = FILE_NAME
    if typ == "tfr":
        meta_file += TFR_META_EXTENSION
    else:
        meta_file += GEN_META_EXTENSION
    meta_info = json.load(open(os.path.join(data_dir, meta_file)))
    result = _get_meta_info(d3=True if X_shape.__len__() > 2 else False, typ=typ)
    assert meta_info == result


def test_write_meta_info_name_error():
    ERROR_NAME_LIST = NAMES_LIST.copy()
    ERROR_NAME_LIST[1] = NAME_0
    with pytest.raises(ValueError, match=f"Too many indexes for patient name '{NAME_0}'!"):
        write_meta_info(save_dir="", file_name=FILE_NAME, labels=LABEL_LIST, names=ERROR_NAME_LIST,
                        names_idx=NAME_IDX_LIST, X_shape=X_SHAPE_1D, typ="tfr")


def test_write_meta_info_typ_error():
    typ = "test"
    with pytest.raises(ValueError, match="Wrong type for meta file, check your configurations!"):
        write_meta_info(save_dir="", file_name=FILE_NAME, labels=LABEL_LIST, names=NAMES_LIST,
                        names_idx=NAME_IDX_LIST, X_shape=X_SHAPE_1D, typ=typ)


@pytest.fixture
def delete_meta_info(data_dir: str):
    yield
    path = glob(os.path.join(data_dir, FILE_NAME + ".*meta"))[0]
    os.remove(path)


def _get_meta_info(d3: bool, typ: str) -> dict:
    if d3:
        meta_info = _base_meta_data(X_shape=list(X_SHAPE_3D)[1:], typ=typ)
        meta_info[f"{FEATURE_X_AXIS_1}_size_dtype"] = "int64"
        meta_info[f"{FEATURE_X_AXIS_2}_size_dtype"] = "int64"
    else:
        meta_info = _base_meta_data(X_shape=list(X_SHAPE_1D)[1:], typ=typ)

    return meta_info


def _base_meta_data(X_shape: list, typ: str) -> dict:
    base_meta = {
        TOTAL_SAMPLES: TOTAL_SAMPLES_,
        f"{FEATURE_X}_shape": X_shape,
        f"{FEATURE_X}_dtype": "float32",
        f"{FEATURE_Y}_dtype": "int64",
        f"{FEATURE_IDX_CUBE}_dtype": "int64",
        f"{FEATURE_WEIGHTS}_dtype": "float32",
        f"{FEATURE_PAT_IDX}_dtype": "int64",
        SAMPLES_PER_NAME: {
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

    if typ == "tfr":
        base_meta[f"{FEATURE_SAMPLES}_dtype"] = "int64"
        base_meta[f"{FEATURE_SPEC}_size_dtype"] = "int64"
        base_meta[f"{FEATURE_PAT_NAME}_seperator"] = PAT_NAME_SEPERATOR
        base_meta[f"{FEATURE_PAT_NAME}_encoding"] = PAT_NAME_ENCODING

    return base_meta
