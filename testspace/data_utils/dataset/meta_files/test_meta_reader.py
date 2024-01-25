import pytest
import os

from glob import glob
from data_utils.dataset.meta_files import get_cw_from_meta, get_meta_files
from configuration.parameter import (
    TFR_FILE_EXTENSION,
)

CW_TEST_DATA = [("1d", [0, 1, 2, 3], ["test_0", "test_1", "test_2", "test_3", "test_4"],
                 {"0": 1.25, "1": 0.625, "2": 2.5, "3": 0.833}),
                ("1d", [0, 1, 3], ["test_0", "test_1", "test_2", "test_3"], {"0": 1.458, "1": 0.729, "3": 1.061}),
                ("3d", [1, 2, 3], ["test_0", "test_2", "test_3", "test_4"], {"1": 0.667, "2": 3.556, "3": 0.821}),
                ("1d", [0, 1, 2, 3, 4], ["test_0", "test_1", "test_2", "test_3", "test_4"],
                 {"0": 1., "1": 0.5, "2": 2., "3": 0.667, "4": 0.0})]


@pytest.mark.parametrize("shape,labels,names,result", CW_TEST_DATA)
def test_cw_from_meta(tfr_data_dir: str, shape: str, labels: list, names: list, result: dict):
    stored_path = glob(os.path.join(tfr_data_dir, shape, "*" + TFR_FILE_EXTENSION))
    cw = get_cw_from_meta(files=stored_path, labels=labels, names=names, dataset_typ="tfr")
    for (cw_l, cw_v), (res_l, res_v) in zip(cw.items(), result.items()):
        assert cw_l == res_l and pytest.approx(cw_v, rel=1e-3) == res_v


def test_cw_from_meta_error():
    with pytest.raises(FileNotFoundError):
        get_cw_from_meta(files=["test"], labels=[], names=[])


GET_META_FILES_DATA = [(["test_0.tfrecords", "test_1.tfrecords"], "tfr", ["test_0.tfrmeta", "test_1.tfrmeta"]),
                       (["test_0.npz", "test_1.npz"], "generator", ["test_0.meta", "test_1.meta"]),
                       (["test.0/data_0.npz", "data_1.npz"], "generator", ["test.0/data_0.meta", "data_1.meta"])]


@pytest.mark.parametrize("paths,dataset_typ,results", GET_META_FILES_DATA)
def test_get_meta_files(paths: list, dataset_typ: str, results: list):
    meta_paths = get_meta_files(paths=paths, typ=dataset_typ)
    assert meta_paths == results


def test_get_meta_files_error():
    with pytest.raises(ValueError, match="Wrong meta file typ to load!"):
        get_meta_files(paths=[], typ="test")
