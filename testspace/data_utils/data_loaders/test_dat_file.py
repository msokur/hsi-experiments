import pytest
import os

import numpy as np
from data_utils.data_loaders.dat_file import DatFile

RED = [255, 0, 0, 255]
GREEN = [0, 255, 0, 255]
BLUE = [0, 0, 255, 255]
BLACK = [0, 0, 0, 255]
NO_ANNO = [0, 0, 0, 0]

RGB_MASK_COLOR = {0: [RED[0:3]],
                  1: [GREEN[0:3]],
                  2: [BLUE[0:3]]}

RGBA_MASK_COLOR = {0: [RED[0:3] + [200]],
                   1: [GREEN[0:3] + [200]],
                   2: [BLUE[0:3] + [200]]}

COMBINED_MASK_COLOR = {0: [RED[0:3]],
                       1: [GREEN[0:3], BLUE[0:3] + [200]]}

ERROR1_MASK_COLOR = {0: [[255, 0]]}

ERROR2_MASK_COLOR = {0: [255, 0, 0]}

ERROR3_MASK_COLOR = {0: [[]]}

WARNING_MASK_COLOR = {0: [[255, 0, 0, 200, 100]]}

LOADER_CONFIG = {"TISSUE_LABELS": {0: "Class0",
                                   1: "Class1",
                                   2: "Class2"},
                 "MASK_DIFF": ["dat.dat", "mask.png"],
                 "WAVE_AREA": 100,
                 "FIRST_NM": 8,
                 "LAST_NM": 100}

RGB_RESULT_BOOL_MASK = [np.array([[True, True, False], [False, False, False], [False, False, False]]),
                        np.array([[False, False, False], [True, False, True], [False, False, False]]),
                        np.array([[False, False, False], [False, False, False], [False, True, True]])]

RGB_RESULT_INDEX_MASK = np.array([[0, 0, -1], [1, -1, 1], [-1, 2, 2]])

RGBA_RESULT_BOOL_MASK = [np.array([[True, False, False], [False, False, False], [False, False, False]]),
                         np.array([[False, False, False], [True, False, False], [False, False, False]]),
                         np.array([[False, False, False], [False, False, False], [False, True, False]])]

RGBA_RESULT_INDEX_MASK = np.array([[0, -1, -1], [1, -1, -1], [-1, 2, -1]])

COMBINED_RESULT_BOOL_MASK = [np.array([[True, True, False], [False, False, False], [False, False, False]]),
                             np.array([[False, False, False], [True, False, True], [False, True, False]])]

COMBINED_RESULT_INDEX_MASK = np.array([[0, 0, -1], [1, -1, 1], [-1, 1, -1]])

PNG_IMG_1 = np.array([[RED, NO_ANNO, NO_ANNO],
                      [NO_ANNO, GREEN, NO_ANNO],
                      [NO_ANNO, NO_ANNO, BLUE],
                      [NO_ANNO, NO_ANNO, NO_ANNO]])

PNG_IMG_2 = np.array([[RED, BLACK, BLACK],
                      [BLACK, GREEN, BLACK],
                      [BLACK, BLACK, BLUE],
                      [BLACK, BLACK, BLACK]])

JPG_IMG = np.array([[[96, 85, 0, 255], [255, 255, 198, 255], [255, 255, 237, 255]],
                    [[255, 255, 198, 255], [151, 148, 133, 255], [251, 251, 255, 255]],
                    [[255, 255, 239, 255], [251, 251, 255, 255], [15, 19, 108, 255]],
                    [[255, 255, 248, 255], [246, 248, 255, 255], [238, 242, 255, 255]]])

MK2_RESULT_MASK = np.array([[NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, GREEN, NO_ANNO, NO_ANNO],
                            [NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, GREEN, GREEN, GREEN, NO_ANNO],
                            [NO_ANNO, RED, NO_ANNO, GREEN, GREEN, GREEN, GREEN, GREEN],
                            [NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, GREEN, GREEN, GREEN, NO_ANNO],
                            [NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, GREEN, NO_ANNO, NO_ANNO],
                            [NO_ANNO, RED, NO_ANNO, NO_ANNO, NO_ANNO, BLUE, NO_ANNO, NO_ANNO],
                            [RED, RED, RED, NO_ANNO, BLUE, BLUE, BLUE, NO_ANNO],
                            [NO_ANNO, RED, NO_ANNO, NO_ANNO, NO_ANNO, BLUE, NO_ANNO, NO_ANNO]])


def GET_MASK_COLOR(typ, config):
    if typ == "RGB":
        config.CONFIG_DATALOADER["MASK_COLOR"] = RGB_MASK_COLOR
    elif typ == "RGBA":
        config.CONFIG_DATALOADER["MASK_COLOR"] = RGBA_MASK_COLOR
    elif typ == "COMBINED":
        config.CONFIG_DATALOADER["MASK_COLOR"] = COMBINED_MASK_COLOR
    elif typ == "ERROR1":
        config.CONFIG_DATALOADER["MASK_COLOR"] = ERROR1_MASK_COLOR
    elif typ == "ERROR2":
        config.CONFIG_DATALOADER["MASK_COLOR"] = ERROR2_MASK_COLOR
    elif typ == "ERROR3":
        config.CONFIG_DATALOADER["MASK_COLOR"] = ERROR3_MASK_COLOR
    elif typ == "WARNING":
        config.CONFIG_DATALOADER["MASK_COLOR"] = WARNING_MASK_COLOR
    else:
        raise ValueError("Found no typ for 'MASK_COLOR'!")
    return config


@pytest.fixture
def mask():
    return np.array([[[255, 0, 0, 255], [255, 0, 0, 128], [0, 0, 0, 255]],
                     [[0, 255, 0, 255], [128, 100, 10, 255], [0, 255, 0, 128]],
                     [[0, 0, 0, 0], [0, 0, 255, 255], [0, 0, 255, 128]]])


INDEXES_GET_BOOL_FROM_MASK_DATA = [(DatFile, "RGB", RGB_RESULT_BOOL_MASK),
                                   (DatFile, "RGBA", RGBA_RESULT_BOOL_MASK),
                                   (DatFile, "COMBINED", COMBINED_RESULT_BOOL_MASK)]


@pytest.mark.parametrize("ext_loader,color,result", INDEXES_GET_BOOL_FROM_MASK_DATA)
def test_indexes_get_bool_from_mask(test_config, mask, ext_loader, color: str, result):
    loader = ext_loader(config=GET_MASK_COLOR(typ=color, config=test_config))
    bool_masks = loader.indexes_get_bool_from_mask(mask=mask)
    for bool_mask, result_mask in zip(bool_masks, result):
        assert np.all(bool_mask == result_mask)


INDEXES_GET_BOOL_FROM_MASK_DATA_ERROR = [(DatFile, "ERROR1",
                                          "Check your configurations in 'MASK_COLOR' for the classification 0! "
                                          "You need a RGB or RGBA value!"),
                                         (DatFile, "ERROR2",
                                          "Check your configurations in 'MASK_COLOR' for the classification 0! "
                                          "Surround your RGB/RGBA value with brackets!"),
                                         (DatFile, "ERROR3",
                                          "Check your configurations in 'MASK_COLOR' for the classification 0! "
                                          "You need a RGB or RGBA value!")]


@pytest.mark.parametrize("ext_loader,error_typ,error", INDEXES_GET_BOOL_FROM_MASK_DATA_ERROR)
def test_indexes_get_bool_from_mask_error(test_config, mask, ext_loader, error_typ: str, error: str):
    loader = ext_loader(config=GET_MASK_COLOR(typ=error_typ, config=test_config))
    with pytest.raises(ValueError, match=error):
        loader.indexes_get_bool_from_mask(mask=mask)


def test_indexes_get_bool_from_mask_warning(test_config, mask):
    loader = DatFile(config=GET_MASK_COLOR(typ="WARNING", config=test_config))
    with pytest.warns(UserWarning, match="To many values in 'MASK_COLOR' for the classification 0! "
                                         "Only the first four will be used."):
        bool_masks = loader.indexes_get_bool_from_mask(mask=mask)

    assert np.all(bool_masks[0] == RGBA_RESULT_BOOL_MASK[0])


SET_MASK_WITH_LABEL_DATA = [(DatFile, "RGB", RGB_RESULT_INDEX_MASK),
                            (DatFile, "RGBA", RGBA_RESULT_INDEX_MASK),
                            (DatFile, "COMBINED", COMBINED_RESULT_INDEX_MASK)]


@pytest.mark.parametrize("ext_loader,color,result", SET_MASK_WITH_LABEL_DATA)
def test_set_mask_with_label(test_config, mask, ext_loader, color: str, result):
    loader = ext_loader(config=GET_MASK_COLOR(typ=color, config=test_config))
    assert np.all(loader.set_mask_with_label(mask=mask) == result)


SPECTRUM_RESULT_4_5_92 = np.full(shape=(4, 5, LOADER_CONFIG["LAST_NM"] - LOADER_CONFIG["FIRST_NM"]),
                                 fill_value=np.arange(start=0.009, stop=0.1, step=0.001, dtype=np.float32))


@pytest.fixture
def dat_path(dat_data_dir: str) -> str:
    return os.path.join(dat_data_dir, "test_dat.dat")


def test_file_read_mask_and_spectrum_only_spectrum(test_config, dat_path: str):
    spectrum, _ = DatFile(config=test_config).file_read_mask_and_spectrum(dat_path)
    assert np.allclose(spectrum, SPECTRUM_RESULT_4_5_92)


def test_file_read_mask_and_spectrum_only_mask(test_config, dat_path: str):
    _, mask = DatFile(config=test_config).file_read_mask_and_spectrum(dat_path)
    assert np.all(mask == PNG_IMG_1)


def test_spectrum_read_from_dat(test_config, dat_path: str):
    cube = DatFile(config=test_config).spectrum_read_from_dat(dat_path=dat_path)
    assert np.allclose(cube, SPECTRUM_RESULT_4_5_92)


MASK_READ_PNG_DATA = [("test_mask.png", PNG_IMG_1),
                      ("test_mask2.png", PNG_IMG_2)]


@pytest.mark.parametrize("img_name,result", MASK_READ_PNG_DATA)
def test_mask_read_png(dat_data_dir: str, img_name, result):
    path = os.path.join(dat_data_dir, img_name)
    assert np.all(DatFile.mask_read(mask_path=path) == result)


MASK_READ_JPG_DATA = ["test_mask.jpg",
                      "test_mask.jpeg",
                      "test_mask.jpe"]


@pytest.mark.parametrize("img_name", MASK_READ_JPG_DATA)
def test_mask_read_jpg(dat_data_dir: str, img_name):
    path = os.path.join(dat_data_dir, img_name)
    with pytest.warns(UserWarning, match="Better use '.png' format. Alpha channel added."):
        DatFile.mask_read(mask_path=path)


FORMAT_ERROR = "Mask format not supported! Only '.png', '.jpeg', '.jpg' or '.jpe' are supported."

MASK_READ_ERROR_DATA = [("test_mask.bmp", FORMAT_ERROR, ValueError),
                        ("test_mask.pbm", FORMAT_ERROR, ValueError),
                        ("test_mask.tiff", FORMAT_ERROR, ValueError),
                        ("", "Mask file not found. Check your configurations!", FileNotFoundError),
                        ("test.png", "Mask file not found. Check your configurations!", FileNotFoundError)]


@pytest.mark.parametrize("img_name,error_msg,error_typ", MASK_READ_ERROR_DATA)
def test_mask_read_error(dat_data_dir: str, img_name, error_msg, error_typ):
    path = os.path.join(dat_data_dir, img_name)
    with pytest.raises(error_typ, match=error_msg):
        DatFile.mask_read(mask_path=path)


MK2_MASK_DATA = [(DatFile, "RGB", MK2_RESULT_MASK),
                 (DatFile, "RGBA", MK2_RESULT_MASK)]


@pytest.mark.parametrize("ext_loader,color,result", MK2_MASK_DATA)
def test_mk2_mask(test_config, dat_data_dir: str, ext_loader, color: str, result):
    loader = ext_loader(config=GET_MASK_COLOR(typ=color, config=test_config))
    mask = loader.mk2_mask(mask_path=os.path.join(dat_data_dir, "test_mask.mk2"),
                           shape=(8, 8))
    assert np.all(mask == result)
