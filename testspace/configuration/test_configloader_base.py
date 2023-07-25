import pytest

from configuration.configloader_base import concat_dict, convert_key_to_int, read_config

DICT_1_DATA = {"MASK_COLOR": {"0": [[255, 0, 0]],
                              "1": [[0, 255, 0]],
                              "2": [[0, 0, 255]]}}

DICT_2_DATA = {"TEST_VALE": "test",
               0: "zero",
               1: "one"}

CONCAT_DICT_RESULT = {"MASK_COLOR": {"0": [[255, 0, 0]],
                                     "1": [[0, 255, 0]],
                                     "2": [[0, 0, 255]]},
                      "TEST_VALE": "test",
                      0: "zero",
                      1: "one"}


def test_concat_dict():
    assert concat_dict(dict1=DICT_1_DATA, dict2=DICT_2_DATA) == CONCAT_DICT_RESULT


CONVERT_KEY_TO_INT_RESULT = {0: [[255, 0, 0]],
                             1: [[0, 255, 0]],
                             2: [[0, 0, 255]]}


def test_convert_key_to_int():
    assert convert_key_to_int(str_dict=DICT_1_DATA["MASK_COLOR"]) == CONVERT_KEY_TO_INT_RESULT


CONVERT_KEY_TO_INT_ERROR_DATA = {"0": "zero",
                                 "one": "one"}


def test_convert_key_to_int_error():
    with pytest.raises(ValueError):
        convert_key_to_int(str_dict=CONVERT_KEY_TO_INT_ERROR_DATA)


def test_read_config(config_data_dir: str, base_config_result):
    assert read_config(file=config_data_dir, section="TEST") == base_config_result
