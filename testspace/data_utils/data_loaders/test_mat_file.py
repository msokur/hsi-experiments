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


def GET_MASK_COLOR(typ: str, config):
    if typ == "NORMAL":
        config.CONFIG_DATALOADER["MASK_COLOR"] = MASK_COLOR
    elif typ == "NORMAL2":
        config.CONFIG_DATALOADER["MASK_COLOR"] = MASK_COLOR2
    elif typ == "COMBINED":
        config.CONFIG_DATALOADER["MASK_COLOR"] = COMBINED_MASK_COLOR
    elif typ == "ERROR1":
        config.CONFIG_DATALOADER["MASK_COLOR"] = ERROR1_MASK_COLOR
    elif typ == "ERROR2":
        config.CONFIG_DATALOADER["MASK_COLOR"] = ERROR2_MASK_COLOR

    return config


@pytest.fixture
def mask():
    return np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 2]])


INDEX_GET_BOOL_FROM_MASK_DATA = [(MatFile, "NORMAL", RESULT_BOOL_MASK),
                                 (MatFile, "NORMAL2", RESULT_BOOL_MASK2),
                                 (MatFile, "COMBINED", RESULT_COMBINED_BOOL_MASK)]


@pytest.mark.parametrize("ext_loader,anno,result", INDEX_GET_BOOL_FROM_MASK_DATA)
def test_indexes_get_bool_from_mask(test_config, mask, ext_loader, anno: str, result):
    loader = ext_loader(config=GET_MASK_COLOR(typ=anno, config=test_config))
    bool_masks = loader.indexes_get_bool_from_mask(mask=mask)
    for bool_mask, result_mask in zip(bool_masks, result):
        assert (bool_mask == result_mask).all()


INDEX_GET_BOOL_FROM_MASK_DATA_ERROR = [(MatFile, "ERROR1",
                                        "Check your configurations for classification 0! "
                                        "Surround your value with brackets!"),
                                       (MatFile, "ERROR2",
                                        "Check your configurations for classification 0! "
                                        "No annotation value is given!")]


@pytest.mark.parametrize("ext_loader,error_typ,error", INDEX_GET_BOOL_FROM_MASK_DATA_ERROR)
def test_indexes_get_bool_from_mask_error(test_config, mask, ext_loader, error_typ: str, error):
    loader = ext_loader(config=GET_MASK_COLOR(typ=error_typ, config=test_config))
    with pytest.raises(ValueError, match=error):
        loader.indexes_get_bool_from_mask(mask=mask)


SET_MASK_WITH_LABEL_DATA = [(MatFile, "NORMAL", RESULT_LABEL_MASK),
                            (MatFile, "NORMAL2", RESULT_LABEL_MASK2),
                            (MatFile, "COMBINED", RESULT_COMBINED_LABEL_MASK)]


@pytest.mark.parametrize("ext_loader,anno,result", SET_MASK_WITH_LABEL_DATA)
def test_set_mask_with_label(test_config, mask, ext_loader, anno, result):
    loader = ext_loader(config=GET_MASK_COLOR(typ=anno, config=test_config))
    assert (loader.set_mask_with_label(mask) == result).all()


def test_file_read_mask_and_spectrum(test_config, mat_data_dir: str):
    test_config.CONFIG_DATALOADER["SPECTRUM"] = "test_spec"
    test_config.CONFIG_DATALOADER["MASK"] = "test_mask"
    loader = MatFile(config=test_config)
    spec, mask = loader.file_read_mask_and_spectrum(os.path.join(mat_data_dir, "test_data.mat"))

    assert (spec == np.full(shape=(3, 3, 10), fill_value=np.arange(start=0, stop=10, step=1))).all()

    assert (mask == RESULT_LABEL_MASK).all()
