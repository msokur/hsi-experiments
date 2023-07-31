import pytest
import numpy as np
import os

from data_utils.data_loaders.mat_file import MatFile

MASK_COLOR = {0: [0],
              1: [1],
              2: [2]}

MASK_COLOR2 = {0: [2],
               1: [0],
               2: [1]}

COMBINED_MASK_COLOR = {0: [0],
                       1: [1, 2]}

ERROR1_MASK_COLOR = {0: 1}
ERROR2_MASK_COLOR = {0: []}

LOADER_CONFIG = {"SPECTRUM": "test_spec",
                 "MASK": "test_mask"}

RESULT_BOOL_MASK = [np.array([[True, True, True], [True, False, True], [True, True, False]]),
                    np.array([[False, False, False], [False, True, False], [False, False, False]]),
                    np.array([[False, False, False], [False, False, False], [False, False, True]])]

RESULT_LABEL_MASK = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 2]])

RESULT_BOOL_MASK2 = [np.array([[False, False, False], [False, False, False], [False, False, True]]),
                     np.array([[True, True, True], [True, False, True], [True, True, False]]),
                     np.array([[False, False, False], [False, True, False], [False, False, False]])]

RESULT_LABEL_MASK2 = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 0]])

RESULT_COMBINED_BOOL_MASK = [np.array([[True, True, True], [True, False, True], [True, True, False]]),
                             np.array([[False, False, False], [False, True, False], [False, False, True]])]

RESULT_COMBINED_LABEL_MASK = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])


def GET_MASK_COLOR(typ: str):
    if typ == "NORMAL":
        LOADER_CONFIG["MASK_COLOR"] = MASK_COLOR
    elif typ == "NORMAL2":
        LOADER_CONFIG["MASK_COLOR"] = MASK_COLOR2
    elif typ == "COMBINED":
        LOADER_CONFIG["MASK_COLOR"] = COMBINED_MASK_COLOR
    elif typ == "ERROR1":
        LOADER_CONFIG["MASK_COLOR"] = ERROR1_MASK_COLOR
    elif typ == "ERROR2":
        LOADER_CONFIG["MASK_COLOR"] = ERROR2_MASK_COLOR

    return LOADER_CONFIG.copy()


@pytest.fixture
def mask():
    return np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 2]])


INDEX_GET_BOOL_FROM_MASK_DATA = [(MatFile(dataloader_config=GET_MASK_COLOR("NORMAL")), RESULT_BOOL_MASK),
                                 (MatFile(dataloader_config=GET_MASK_COLOR("NORMAL2")), RESULT_BOOL_MASK2),
                                 (MatFile(dataloader_config=GET_MASK_COLOR("COMBINED")), RESULT_COMBINED_BOOL_MASK)]


@pytest.mark.parametrize("loader,result", INDEX_GET_BOOL_FROM_MASK_DATA)
def test_indexes_get_bool_from_mask(mask, loader, result):
    bool_masks = loader.indexes_get_bool_from_mask(mask=mask)
    for bool_mask, result_mask in zip(bool_masks, result):
        assert (bool_mask == result_mask).all()


INDEX_GET_BOOL_FROM_MASK_DATA_ERROR = [(MatFile(dataloader_config=GET_MASK_COLOR("ERROR1")),
                                        "Check your configurations for classification 0! "
                                        "Surround your value with brackets!"),
                                       (MatFile(dataloader_config=GET_MASK_COLOR("ERROR2")),
                                        "Check your configurations for classification 0! "
                                        "No annotation value is given!")]


@pytest.mark.parametrize("loader,error", INDEX_GET_BOOL_FROM_MASK_DATA_ERROR)
def test_indexes_get_bool_from_mask_error(mask, loader, error):
    with pytest.raises(ValueError, match=error):
        loader.indexes_get_bool_from_mask(mask=mask)


SET_MASK_WITH_LABEL_DATA = [(MatFile(dataloader_config=GET_MASK_COLOR("NORMAL")), RESULT_LABEL_MASK),
                            (MatFile(dataloader_config=GET_MASK_COLOR("NORMAL2")), RESULT_LABEL_MASK2),
                            (MatFile(dataloader_config=GET_MASK_COLOR("COMBINED")), RESULT_COMBINED_LABEL_MASK)]


@pytest.mark.parametrize("loader,result", SET_MASK_WITH_LABEL_DATA)
def test_set_mask_with_label(mask, loader, result):
    assert (loader.set_mask_with_label(mask) == result).all()


def test_file_read_mask_and_spectrum(mat_data_dir: str):
    loader = MatFile(dataloader_config=LOADER_CONFIG)
    spec, mask = loader.file_read_mask_and_spectrum(os.path.join(mat_data_dir, "test_data.mat"))

    assert (spec == np.full(shape=(3, 3, 10), fill_value=np.arange(start=0, stop=10, step=1))).all()

    assert (mask == RESULT_LABEL_MASK).all()
