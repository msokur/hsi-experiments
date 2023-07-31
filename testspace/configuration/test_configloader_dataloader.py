import pytest

from configuration.configloader_dataloader import read_dataloader_config, split_label_data, set_parameter


def test_read_dataloader_config(dataloader_data_dir: str, dataloader_result: dict):
    assert read_dataloader_config(file=dataloader_data_dir, section="DATALOADER") == dataloader_result


SPLIT_LABEL_DATA_DATA = {0: {"MASK": [[255, 255, 0]], "LABEL": "Nerve", "COLOR": "yellow"},
                         1: {"MASK": [[0, 0, 255]], "LABEL": "Tumor", "COLOR": "blue"},
                         2: {"MASK": [[255, 0, 0]], "LABEL": "Parotis", "COLOR": "red"}}

SPLIT_LABEL_DATA_RESULT = {"MASK": {0: [[255, 255, 0]], 1: [[0, 0, 255]], 2: [[255, 0, 0]]},
                           "LABEL": {0: "Nerve", 1: "Tumor", 2: "Parotis"},
                           "COLOR": {0: "yellow", 1: "blue", 2: "red"}}


def test_split_label_data():
    assert split_label_data(label_data=SPLIT_LABEL_DATA_DATA) == SPLIT_LABEL_DATA_RESULT


SET_PARAMETER_DATA_RESULT = {"TEST0": {0: 0, 1: 2},
                             "TEST1": {0: 1, 1: 3}}

SET_PARAMETER_DATA = [(SPLIT_LABEL_DATA_RESULT,
                       {"MASK_COLOR": {0: [[255, 255, 0]], 1: [[0, 0, 255]], 2: [[255, 0, 0]]},
                        "TISSUE_LABELS": {0: "Nerve", 1: "Tumor", 2: "Parotis"},
                        "PLOT_COLORS": {0: "yellow", 1: "blue", 2: "red"}}),
                      (SET_PARAMETER_DATA_RESULT, SET_PARAMETER_DATA_RESULT)]

SET_PARAMS = {"MASK": "MASK_COLOR", "LABEL": "TISSUE_LABELS", "COLOR": "PLOT_COLORS"}


@pytest.mark.parametrize("data,result", SET_PARAMETER_DATA)
def test_set_parameter(data: dict, result: dict):
    assert set_parameter(configuration={}, params=data, replace_dict=SET_PARAMS, origin_label="LABEL_DATA") == result


def test_set_parameter_error():
    with pytest.raises(ValueError, match="Key 'TEST0' from LABEL is already in dataloader configs. Please rename key!"):
        set_parameter(configuration={"TEST0": 1}, params=SET_PARAMETER_DATA_RESULT, replace_dict=SET_PARAMS,
                      origin_label="LABEL")
