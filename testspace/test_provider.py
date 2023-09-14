import pytest
import os
import numpy as np

from provider import get_trainer, get_data_loader, get_whole_analog_of_data_loader, get_evaluation, get_smoother, \
    get_scaler, get_pixel_detection, get_cross_validator, get_extension_loader, get_data_archive

from trainers.trainer_tuner import TrainerTuner
from trainers.trainer_easy import TrainerEasy
from trainers.trainer_easy_several_outputs import TrainerEasySeveralOutputs

from data_utils.data_loaders.data_loader import DataLoader
from data_utils.data_loaders.data_loader_whole import DataLoaderWhole

from evaluation.evaluation_binary import EvaluationBinary
from evaluation.evaluation_multiclass import EvaluationMulticlass

from data_utils.smoothing import MedianFilter, GaussianFilter

from data_utils.scaler import NormalizerScaler, StandardScaler, StandardScalerTransposed

from data_utils.border import detect_border, detect_core

from data_utils.data_archive import DataArchiveZARR, DataArchiveNPZ

from cross_validators.cross_validator_normal import CrossValidationNormal

from data_utils.data_loaders.dat_file import DatFile
from data_utils.data_loaders.mat_file import MatFile

GET_TRAINER_DATA = [("Tuner", "trainer_tuner", TrainerTuner),
                    ("Easy", "trainer_easy", TrainerEasy),
                    ("SeveralOutput", "trainer_multiclass", TrainerEasySeveralOutputs)]


@pytest.mark.parametrize("typ,model_name,result", GET_TRAINER_DATA)
def test_get_trainer(typ, model_name, result):
    trainer = get_trainer(typ=typ, data_archive=DataArchiveNPZ(), config_trainer={}, config_paths={},
                          config_dataloader={}, model_name=model_name)

    assert isinstance(trainer, result)


def test_get_trainer_error():
    with pytest.raises(ValueError, match="Error! No corresponding Trainer for test"):
        get_trainer(typ="test")


GET_DATA_LOADER_DATA = [("normal", DataLoader),
                        ("whole", DataLoaderWhole)]


@pytest.mark.parametrize("typ,result", GET_DATA_LOADER_DATA)
def test_get_data_loader(typ, result):
    loader = get_data_loader(typ=typ, data_archive=DataArchiveNPZ(),
                             config_dataloader={"FILE_EXTENSION": ".dat"}, config_paths={})

    assert isinstance(loader, result)


def test_data_loader_error():
    with pytest.raises(ValueError, match="Error! No corresponding Data Loader for test"):
        get_data_loader(typ="test")


GET_WHOLE_ANALOG_DATA = [("colon", "colon_whole"),
                         ("bea_brain", "bea_brain_whole"),
                         ("bea_eso", "bea_eso_whole"),
                         ("bea_colon", "bea_colon_whole"),
                         ("hno", "hno_whole")]


@pytest.mark.parametrize("typ,result", GET_WHOLE_ANALOG_DATA)
def test_get_whole_analog_of_data_loader(typ, result):
    assert get_whole_analog_of_data_loader(original_database=typ) == result


def test_get_whole_analog_of_data_loader_error():
    with pytest.raises(ValueError, match="We didn't found an analog whole database for test"):
        get_whole_analog_of_data_loader(original_database="test")


GET_EVALUATION_DATA = [([0, 1], EvaluationBinary),
                       ([0, 1, 2], EvaluationMulticlass)]


@pytest.mark.parametrize("labels,result", GET_EVALUATION_DATA)
def test_get_evaluation(labels, result):
    evaluation = get_evaluation(labels=labels)

    assert isinstance(evaluation, result)

    os.rmdir(path=evaluation.save_evaluation_folder)


def test_get_evaluation_error():
    with pytest.raises(ValueError, match="Error! No corresponding evaluation for labels length < 2"):
        get_evaluation(labels=[0], name="test")


GET_SMOOTHER_DATA = [("median_filter", "/work/folder", 5, MedianFilter),
                     ("gaussian_filter", "/work/folder2", 2, GaussianFilter)]


@pytest.mark.parametrize("typ,path,size,result", GET_SMOOTHER_DATA)
def test_get_smoother(typ, path, size, result):
    smoother = get_smoother(typ=typ, path=path, size=size)

    assert isinstance(smoother, result)


GET_SCALER_DATA = [("l2_norm", NormalizerScaler),
                   ("svn", StandardScaler),
                   ("svn_T", StandardScalerTransposed)]


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
def test_get_scaler(typ, result):
    path = "scaler_test"
    scaler = get_scaler(typ=typ, preprocessed_path=path, data_archive=DataArchiveNPZ())

    assert isinstance(scaler, result)


def test_get_scaler_error():
    with pytest.raises(ValueError, match="Error! No corresponding scaler for test"):
        get_scaler(typ="test")


GET_PIXEL_DETECTION_DATA = [("detect_border", detect_border),
                            ("detect_core", detect_core)]


@pytest.mark.parametrize("typ,result", GET_PIXEL_DETECTION_DATA)
def test_get_pixel_detection(typ, result):
    assert get_pixel_detection(typ=typ) == result


def test_get_pixel_detection_error():
    with pytest.raises(ValueError, match="Error! No corresponding pixel detection for test"):
        get_pixel_detection(typ="test")


GET_CROSS_VALIDATION_DATA = [("normal", CrossValidationNormal)]


@pytest.mark.parametrize("typ,result", GET_CROSS_VALIDATION_DATA)
def test_get_cross_validation(typ, result):
    assert isinstance(get_cross_validator(typ=typ), result)


def test_get_cross_validation_error():
    with pytest.raises(ValueError, match="Error! No corresponding Cross validator for test"):
        get_cross_validator(typ="test")


GET_EXTENSION_LOADER_DATA = [(".dat", DatFile),
                             (".mat", MatFile)]


@pytest.mark.parametrize("typ,result", GET_EXTENSION_LOADER_DATA)
def test_get_extension_loader(typ, result):
    assert isinstance(get_extension_loader(typ=typ, dataloader_config={}), result)


def test_get_extension_loader_error():
    with pytest.raises(ValueError, match="Error! No corresponding file extension for test"):
        get_extension_loader(typ="test")


GET_DATA_ARCHIVE = [("npz", DataArchiveNPZ),
                    ("zarr", DataArchiveZARR)]


@pytest.mark.parametrize("typ,archive", GET_DATA_ARCHIVE)
def test_get_data_archive(typ: str, archive):
    assert isinstance(get_data_archive(typ=typ), archive)


def test_get_data_archive_error():
    with pytest.raises(ValueError, match="Error! No corresponding data archive for test"):
        get_data_archive(typ="test")
