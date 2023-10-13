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

JPG_IMG = np.array([[[89, 82, 14, 255], [255, 255, 193, 255], [250, 252, 255, 255]],
                    [[255, 255, 193, 255], [160, 153, 85, 255], [252, 254, 255, 255]],
                    [[254, 253, 255, 255], [254, 253, 255, 255], [11, 15, 138, 255]],
                    [[254, 253, 255, 255], [252, 251, 255, 255], [236, 240, 255, 255]]])

MK2_RESULT_MASK = np.array([[NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, GREEN, NO_ANNO, NO_ANNO],
                            [NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, GREEN, GREEN, GREEN, NO_ANNO],
                            [NO_ANNO, RED, NO_ANNO, GREEN, GREEN, GREEN, GREEN, GREEN],
                            [NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, GREEN, GREEN, GREEN, NO_ANNO],
                            [NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, NO_ANNO, GREEN, NO_ANNO, NO_ANNO],
                            [NO_ANNO, RED, NO_ANNO, NO_ANNO, NO_ANNO, BLUE, NO_ANNO, NO_ANNO],
                            [RED, RED, RED, NO_ANNO, BLUE, BLUE, BLUE, NO_ANNO],
                            [NO_ANNO, RED, NO_ANNO, NO_ANNO, NO_ANNO, BLUE, NO_ANNO, NO_ANNO]])


def GET_MASK_COLOR(typ):
    if typ == "RGB":
        LOADER_CONFIG["MASK_COLOR"] = RGB_MASK_COLOR
    elif typ == "RGBA":
        LOADER_CONFIG["MASK_COLOR"] = RGBA_MASK_COLOR
    elif typ == "COMBINED":
        LOADER_CONFIG["MASK_COLOR"] = COMBINED_MASK_COLOR
    elif typ == "ERROR1":
        LOADER_CONFIG["MASK_COLOR"] = ERROR1_MASK_COLOR
    elif typ == "ERROR2":
        LOADER_CONFIG["MASK_COLOR"] = ERROR2_MASK_COLOR
    elif typ == "ERROR3":
        LOADER_CONFIG["MASK_COLOR"] = ERROR3_MASK_COLOR
    elif typ == "WARNING":
        LOADER_CONFIG["MASK_COLOR"] = WARNING_MASK_COLOR
    else:
        raise ValueError("Found no typ for 'MASK_COLOR'!")
    return LOADER_CONFIG.copy()


@pytest.fixture
def mask():
    return np.array([[[255, 0, 0, 255], [255, 0, 0, 128], [0, 0, 0, 255]],
                     [[0, 255, 0, 255], [128, 100, 10, 255], [0, 255, 0, 128]],
                     [[0, 0, 0, 0], [0, 0, 255, 255], [0, 0, 255, 128]]])


INDEXES_GET_BOOL_FROM_MASK_DATA = [(DatFile(dataloader_config=GET_MASK_COLOR("RGB")), RGB_RESULT_BOOL_MASK),
                                   (DatFile(dataloader_config=GET_MASK_COLOR("RGBA")), RGBA_RESULT_BOOL_MASK),
                                   (DatFile(dataloader_config=GET_MASK_COLOR("COMBINED")), COMBINED_RESULT_BOOL_MASK)]


@pytest.mark.parametrize("loader,result", INDEXES_GET_BOOL_FROM_MASK_DATA)
def test_indexes_get_bool_from_mask(mask, loader, result):
    bool_masks = loader.indexes_get_bool_from_mask(mask=mask)
    for bool_mask, result_mask in zip(bool_masks, result):
        assert (bool_mask == result_mask).all()


INDEXES_GET_BOOL_FROM_MASK_DATA_ERROR = [(DatFile(dataloader_config=GET_MASK_COLOR("ERROR1")),
                                          "Check your configurations in 'MASK_COLOR' for the classification 0! "
                                          "You need a RGB or RGBA value!"),
                                         (DatFile(dataloader_config=GET_MASK_COLOR("ERROR2")),
                                          "Check your configurations in 'MASK_COLOR' for the classification 0! "
                                          "Surround your RGB/RGBA value with brackets!"),
                                         (DatFile(dataloader_config=GET_MASK_COLOR("ERROR3")),
                                          "Check your configurations in 'MASK_COLOR' for the classification 0! "
                                          "You need a RGB or RGBA value!")]


@pytest.mark.parametrize("loader,error", INDEXES_GET_BOOL_FROM_MASK_DATA_ERROR)
def test_indexes_get_bool_from_mask_error(mask, loader, error):
    with pytest.raises(ValueError, match=error):
        loader.indexes_get_bool_from_mask(mask=mask)


def test_indexes_get_bool_from_mask_warning(mask):
    loader = DatFile(dataloader_config=GET_MASK_COLOR("WARNING"))
    with pytest.warns(UserWarning, match="To many values in 'MASK_COLOR' for the classification 0! "
                                         "Only the first four will be used."):
        bool_masks = loader.indexes_get_bool_from_mask(mask=mask)

    assert (bool_masks[0] == RGBA_RESULT_BOOL_MASK[0]).all()


SET_MASK_WITH_LABEL_DATA = [(DatFile(dataloader_config=GET_MASK_COLOR("RGB")), RGB_RESULT_INDEX_MASK),
                            (DatFile(dataloader_config=GET_MASK_COLOR("RGBA")), RGBA_RESULT_INDEX_MASK),
                            (DatFile(dataloader_config=GET_MASK_COLOR("COMBINED")), COMBINED_RESULT_INDEX_MASK)]


@pytest.mark.parametrize("loader,result", SET_MASK_WITH_LABEL_DATA)
def test_set_mask_with_label(mask, loader, result):
    print(loader.set_mask_with_label(mask=mask))
    assert (loader.set_mask_with_label(mask=mask) == result).all()


SPECTRUM_RESULT_4_5_92 = np.full(shape=(4, 5, LOADER_CONFIG["LAST_NM"] - LOADER_CONFIG["FIRST_NM"]),
                                 fill_value=np.arange(start=0.009, stop=0.1, step=0.001, dtype=np.float32))


@pytest.fixture
def dat_path(dat_data_dir: str) -> str:
    return os.path.join(dat_data_dir, "test_dat.dat")


def test_file_read_mask_and_spectrum_only_spectrum(dat_path: str):
    spectrum, _ = DatFile(dataloader_config=LOADER_CONFIG).file_read_mask_and_spectrum(dat_path)
    assert np.allclose(spectrum, SPECTRUM_RESULT_4_5_92)


def test_file_read_mask_and_spectrum_only_mask(dat_path: str):
    _, mask = DatFile(dataloader_config=LOADER_CONFIG).file_read_mask_and_spectrum(dat_path)
    assert (mask == PNG_IMG_1).all()


def test_spectrum_read_from_dat(dat_path: str):
    cube = DatFile(dataloader_config=LOADER_CONFIG).spectrum_read_from_dat(dat_path=dat_path)
    assert np.allclose(cube, SPECTRUM_RESULT_4_5_92)


MASK_READ_PNG_DATA = [("test_mask.png", PNG_IMG_1),
                      ("test_mask2.png", PNG_IMG_2)]


@pytest.mark.parametrize("img_name,result", MASK_READ_PNG_DATA)
def test_mask_read_png(dat_data_dir: str, img_name, result):
    path = os.path.join(dat_data_dir, img_name)
    assert (DatFile.mask_read(mask_path=path) == result).all()


MASK_READ_JPG_DATA = [("test_mask.jpg", JPG_IMG),
                      ("test_mask.jpeg", JPG_IMG),
                      ("test_mask.jpe", JPG_IMG)]


@pytest.mark.parametrize("img_name,result", MASK_READ_JPG_DATA)
def test_mask_read_jpg(dat_data_dir: str, img_name, result):
    path = os.path.join(dat_data_dir, img_name)
    with pytest.warns(UserWarning, match="Better use '.png' format. Alpha channel added."):
        img = DatFile.mask_read(mask_path=path)
    print(img)
    assert (img == result).all()


FORMAT_ERROR = "Mask format not supported! Only '.png', '.jpeg', '.jpg' or '.jpe' are supported."

MASK_READ_ERROR_DATA = [("test_mask.bmp", FORMAT_ERROR),
                        ("test_mask.pbm", FORMAT_ERROR),
                        ("test_mask.tiff", FORMAT_ERROR),
                        ("", "Mask file not found. Check your configurations!")]


@pytest.mark.parametrize("img_name,error_msg", MASK_READ_ERROR_DATA)
def test_mask_read_error(dat_data_dir: str, img_name, error_msg):
    path = os.path.join(dat_data_dir, img_name)
    with pytest.raises(ValueError, match=error_msg):
        DatFile.mask_read(mask_path=path)


MK2_MASK_DATA = [(DatFile(dataloader_config=GET_MASK_COLOR("RGB")), MK2_RESULT_MASK),
                 (DatFile(dataloader_config=GET_MASK_COLOR("RGBA")), MK2_RESULT_MASK)]


@pytest.mark.parametrize("loader,result", MK2_MASK_DATA)
def test_mk2_mask(dat_data_dir: str, loader, result):
    mask = loader.mk2_mask(mask_path=os.path.join(dat_data_dir, "test_mask.mk2"),
                           shape=(8, 8))
    print(mask)
    assert (mask == result).all()
