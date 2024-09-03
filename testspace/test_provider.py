import pytest
import os
import numpy as np
from shutil import rmtree

from provider import (
    get_trainer,
    get_data_loader,
    get_cube_loader,
    get_annotation_mask_loader,
    get_whole_analog_of_data_loader,
    get_evaluation, get_smoother,
    get_scaler,
    get_pixel_detection,
    get_cross_validator,
    get_extension_loader,
    get_data_storage
)

from trainers.trainer_tuner import TrainerTuner
from trainers.trainer_binary import TrainerBinary

from data_utils.data_loading.data_loader import (
    DataLoaderNormal,
    DataLoaderFolder
)

from data_utils. data_loaders.data_loader import DataLoader

from data_utils.data_loaders.data_loader_whole import DataLoaderWhole

from data_utils.data_loading.annotation_mask_loader import (
    MatAnnotationMask,
    Mk2AnnotationMask,
    PNGAnnotationMask
)

from data_utils.data_loading.cube_loader import (
    MatCube,
    DatCube
)

from evaluation.evaluation_binary import EvaluationBinary
from evaluation.evaluation_multiclass import EvaluationMulticlass

from data_utils.smoothing import (
    MedianFilter,
    GaussianFilter
)

from data_utils.scaler import (
    NormalizerScaler,
    StandardScalerTransposed
)

from data_utils.border import (
    detect_border,
    detect_core
)

from data_utils.data_storage import (
    DataStorageZARR,
    DataStorageNPZ
)

from cross_validators.cross_validator_base import CrossValidatorBase

from data_utils.data_loaders.dat_file import DatFile
from data_utils.data_loaders.mat_file import MatFile

GET_TRAINER_DATA = [("Tuner", "trainer_tuner", TrainerTuner),
                    ("Normal", "trainer_normal", TrainerBinary)]


@pytest.mark.parametrize("typ,log_dir,result", GET_TRAINER_DATA)
def test_get_trainer(test_config, typ, log_dir, result):
    trainer = get_trainer(typ=typ, config=test_config, data_storage=DataStorageNPZ(), log_dir=log_dir)

    if os.path.exists(log_dir):
        rmtree(log_dir)

    assert isinstance(trainer, result)


def test_get_trainer_error(test_config):
    with pytest.raises(ValueError, match="Error! No corresponding Trainer for test"):
        get_trainer(typ="test", config=test_config)


GET_DATA_LOADER_DATA = [("normal", DataLoaderNormal, {"cube_loader": DatCube, "mask_loader": PNGAnnotationMask}),
                        ("folder", DataLoaderFolder, {"cube_loader": DatCube, "mask_loader": PNGAnnotationMask}),
                        ("old", DataLoader, {}),
                        ("whole", DataLoaderWhole, {})]


@pytest.mark.parametrize("typ,result,kwargs", GET_DATA_LOADER_DATA)
def test_get_data_loader(test_config, typ, result, kwargs):
    for loader in kwargs.values():
        loader(test_config)
    loader = get_data_loader(typ=typ, config=test_config, data_storage=DataStorageNPZ(), **kwargs)

    assert isinstance(loader, result)


def test_data_loader_error(test_config):
    with pytest.raises(ValueError, match="Error! No corresponding Data Loader for test"):
        get_data_loader(typ="test", config=test_config)


GET_CUBE_LOADER_DATA = [(".dat", DatCube),
                        (".mat", MatCube)]


@pytest.mark.parametrize("typ,result", GET_CUBE_LOADER_DATA)
def test_get_cube_loader(test_config, typ: str, result):
    cube_loader = get_cube_loader(typ=typ,
                                  config=test_config)

    assert isinstance(cube_loader, result)


def test_get_cube_loader_error(test_config):
    with pytest.raises(ValueError, match="Error! No corresponding cube loader for test"):
        get_cube_loader(typ="test",
                        config=test_config)


GET_MASK_LOADER_DATA = [(".png", PNGAnnotationMask),
                        (".mk2", Mk2AnnotationMask),
                        (".mat", MatAnnotationMask)]


@pytest.mark.parametrize("typ,result", GET_MASK_LOADER_DATA)
def test_get_annotation_maks_loader(test_config, typ: str, result):
    mask_loader = get_annotation_mask_loader(typ=typ,
                                             config=test_config)

    assert isinstance(mask_loader, result)


def test_get_annotation_maks_loader_error(test_config):
    with pytest.raises(ValueError, match="Error! No corresponding annotation mask loader for test"):
        get_annotation_mask_loader(typ="test",
                                   config=test_config)


GET_WHOLE_ANALOG_DATA = [("normal", "whole")]


@pytest.mark.parametrize("typ,result", GET_WHOLE_ANALOG_DATA)
def test_get_whole_analog_of_data_loader(typ, result):
    assert get_whole_analog_of_data_loader(original_database=typ) == result


def test_get_whole_analog_of_data_loader_error():
    with pytest.raises(ValueError, match="We didn't found an analog whole database for test"):
        get_whole_analog_of_data_loader(original_database="test")


GET_EVALUATION_DATA = [([0, 1], EvaluationBinary),
                       ([0, 1, 2], EvaluationMulticlass)]


@pytest.fixture
def delete_result_folder():
    yield
    if os.path.exists("results"):
        rmtree("results")


@pytest.mark.parametrize("labels,result", GET_EVALUATION_DATA)
def test_get_evaluation(test_config, delete_result_folder, labels, result):
    evaluation = get_evaluation(labels=labels, config=test_config)

    assert isinstance(evaluation, result)


def test_get_evaluation_error(test_config):
    with pytest.raises(ValueError, match="Error! No corresponding evaluation for labels length < 2"):
        get_evaluation(labels=[0], config=test_config)


GET_SMOOTHER_DATA = [("median_filter", MedianFilter),
                     ("gaussian_filter", GaussianFilter)]


@pytest.mark.parametrize("typ,result", GET_SMOOTHER_DATA)
def test_get_smoother(test_config, typ, result):
    smoother = get_smoother(typ=typ, config=test_config)

    assert isinstance(smoother, result)


GET_SCALER_DATA = [("l2_norm", NormalizerScaler),
                   ("svn", StandardScalerTransposed)]


@pytest.fixture
def scaler_path():
    path = "scaler_test"
    if not os.path.exists(path):
        os.mkdir(path=path)
    np.savez(os.path.join(path, "test.npz"),
             **{"X": [[0, 1, 2]], "y": [0], "indexes_in_datacube": [(0, 0)]})
    yield
    if os.path.exists(os.path.join(path, "scaler.scaler")):
        os.remove(path=os.path.join(path, "scaler.scaler"))
    os.remove(path=os.path.join(path, "test.npz"))
    os.rmdir(path=path)


@pytest.mark.usefixtures("scaler_path")
@pytest.mark.parametrize("typ,result", GET_SCALER_DATA)
def test_get_scaler(test_config, typ, result):
    path = "scaler_test"
    scaler = get_scaler(typ=typ, config=test_config, preprocessed_path=path, data_storage=DataStorageNPZ())

    assert isinstance(scaler, result)


def test_get_scaler_error(test_config):
    with pytest.raises(ValueError, match="Error! No corresponding scaler for test"):
        get_scaler(typ="test", config=test_config)


GET_PIXEL_DETECTION_DATA = [("detect_border", detect_border),
                            ("detect_core", detect_core)]


@pytest.mark.parametrize("typ,result", GET_PIXEL_DETECTION_DATA)
def test_get_pixel_detection(typ, result):
    assert get_pixel_detection(typ=typ) == result


def test_get_pixel_detection_error():
    with pytest.raises(ValueError, match="Error! No corresponding pixel detection for test"):
        get_pixel_detection(typ="test")


GET_CROSS_VALIDATION_DATA = [("normal", CrossValidatorBase)]


@pytest.mark.parametrize("typ,result", GET_CROSS_VALIDATION_DATA)
def test_get_cross_validation(test_config, typ, result):
    assert isinstance(get_cross_validator(typ=typ, config=test_config), result)


def test_get_cross_validation_error(test_config):
    with pytest.raises(ValueError, match="Error! No corresponding Cross validator for test"):
        get_cross_validator(typ="test", config=test_config)


GET_EXTENSION_LOADER_DATA = [(".dat", DatFile),
                             (".mat", MatFile)]


@pytest.mark.parametrize("typ,result", GET_EXTENSION_LOADER_DATA)
def test_get_extension_loader(test_config, typ, result):
    assert isinstance(get_extension_loader(typ=typ, config=test_config), result)


def test_get_extension_loader_error(test_config):
    with pytest.raises(ValueError, match="Error! No corresponding file extension for test"):
        get_extension_loader(typ="test", config=test_config)


GET_DATA_STORAGE = [("npz", DataStorageNPZ),
                    ("zarr", DataStorageZARR)]


@pytest.mark.parametrize("typ,data_storage", GET_DATA_STORAGE)
def test_get_data_storage(typ: str, data_storage):
    assert isinstance(get_data_storage(typ=typ), data_storage)


def test_get_data_storage_error():
    with pytest.raises(ValueError, match="Error! No corresponding data storage for test"):
        get_data_storage(typ="test")
