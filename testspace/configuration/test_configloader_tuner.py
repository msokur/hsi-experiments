import pytest

from configuration.configloader_tuner import get_tuner, get_new_dict

GET_TUNER_DATA = ["RandomSearch",
                  "BayesianOptimization",
                  "Hyperband"]


@pytest.mark.parametrize("tuner_name", GET_TUNER_DATA)
def test_get_tuner(tuner: dict, tuner_name: str):
    assert get_tuner(tuner=tuner_name, file="test_file", section="test_section") == tuner[tuner_name]


def test_get_tuner_error(tuner: dict):
    with pytest.raises(ValueError, match="In file 'test_file', section 'test_section' 'TUNER' was wrongly written = "
                                         "doesn't correspond to any of 'RandomSearch', 'BayesianOptimization' or "
                                         "'Hyperband'"):
        get_tuner(tuner="tuner_name", file="test_file", section="test_section")


def get_new_dict_result(load_list: list, load_dict: dict) -> dict:
    return {k: v for k, v in load_dict.items() if k in load_list}


GET_NEW_DICT_DATA = [("Optimizer", ["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rms", "sgd"]),
                     ("Optimizer", ["adadelta", "adagrad", "adam", "ftrl", "nadam"]),
                     ("Activation", ["relu", "tanh", "selu", "exponential", "elu"])]


@pytest.mark.parametrize("name,load_list", GET_NEW_DICT_DATA)
def test_get_new_dict(name: str, load_list: list, optimizer: dict, activation: dict):
    if name == "Optimizer":
        result = get_new_dict_result(load_list=load_list, load_dict=optimizer)
    else:
        result = get_new_dict_result(load_list=load_list, load_dict=activation)

    assert get_new_dict(load_list=load_list, name=name) == result


def test_get_new_dict_warning(optimizer: dict):
    with pytest.warns(UserWarning, match="For 'Optimizer' no values given. Return all possible values!"):
        result = get_new_dict(load_list=[], name="Optimizer")

    assert result == optimizer


def test_get_new_dict_error():
    with pytest.raises(ValueError, match="Key: 'test' in Optimizer not available!"):
        get_new_dict(load_list=["adadelta", "test"], name="Optimizer")
