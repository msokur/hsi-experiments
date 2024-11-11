from typing import Tuple

import pytest

from configuration.configloader_trainer import read_trainer_config, get_metric_objects, get_metric, get_model


@pytest.fixture(params=[("NORMAL", True, "paper_model", "multi", [("F1_score", [0])]),
                        ("NORMAL", False, "paper_model", "binary", [("F1_score", [0])]),
                        ("TUNER", True, "inception_model", "multi", [("F1_score", [0])]),
                        ("TUNER", False, "inception_model", "binary", [("F1_score", [0])])])
def trainer_result(request, trainer_normal_base: dict, trainer_tuner_base: dict, model_normal: dict, model_tuner,
                   metric: dict) -> Tuple[str, bool, list, dict]:
    param = request.param
    if param[0] == "NORMAL":
        result = trainer_normal_base.copy()
        model = model_normal
    else:
        result = trainer_tuner_base.copy()
        model = model_tuner

    result["MODEL"] = model["3D" if param[1] else "1D"][param[2]]
    for met, obj in param[4]:
        for idx in obj:
            result["CUSTOM_OBJECTS"][idx]["metric"] = metric[param[3]][met]
        if met not in result["CUSTOM_OBJECTS_LOAD"]:
            result["CUSTOM_OBJECTS_LOAD"].update({met: metric[param[3]][met]})
    return param[0], param[1], [0, 1, 2] if param[3] == "multi" else [0, 1], result


def test_read_trainer_without_model(trainer_data_dir: str, trainer_result: tuple):
    section, d3, classes, result = trainer_result
    trainer = read_trainer_config(file=trainer_data_dir, section=section, d3=d3, classes=classes)
    print(result)
    trainer.pop("MODEL")
    result.pop("MODEL")
    assert trainer == result


def test_read_trainer_only_model(trainer_data_dir: str, trainer_result: tuple):
    section, d3, classes, result = trainer_result
    trainer_normal = read_trainer_config(file=trainer_data_dir, section=section, d3=d3, classes=classes)
    trainer_model = trainer_normal.pop("MODEL")
    result_model = result.pop("MODEL")
    if section == "NORMAL":
        assert trainer_model == result_model
    else:
        assert trainer_model == result_model


@pytest.mark.parametrize("d3", [True, False])
def test_read_trainer_error(trainer_data_dir: str, d3: bool):
    with pytest.raises(ValueError, match="Model 'test', not implemented!"):
        read_trainer_config(file=trainer_data_dir, section="ERROR", d3=d3, classes=[0, 1, 2])


@pytest.fixture(params=[(True, "Normal"),
                        (False, "Normal"),
                        (True, "Tuner"),
                        (False, "Tuner")])
def get_model_result(request, model_normal: dict, model_tuner: dict) -> Tuple[bool, str, dict]:
    param = request.param
    if param[1] == "Tuner":
        model = model_tuner
    else:
        model = model_normal

    return param[0], param[1], model["3D" if param[0] else "1D"]


GET_METRIC_OBJECTS_DATA = {"F1_score": [{"name": "f1_score_weighted", "average": "weighted"},
                                        {"name": "f1_score_micro", "average": "micro"}]}


@pytest.fixture(params=[(True, "multi"),
                        (False, "binary")])
def get_metric_objects_result(request, metric: dict) -> Tuple[tuple, bool]:
    param = request.param
    result_obj = {0: {"metric": metric[param[1]]["F1_score"],
                      "args": {"name": "f1_score_weighted", "average": "weighted"}},
                  1: {"metric": metric[param[1]]["F1_score"],
                      "args": {"name": "f1_score_micro", "average": "micro"}}}
    result_load = {"F1_score": metric[param[1]]["F1_score"]}

    return (result_obj, result_load), param[0]


def test_get_metric_objects(get_metric_objects_result: tuple):
    result, multiclass = get_metric_objects_result
    metrics_obj = get_metric_objects(metrics=GET_METRIC_OBJECTS_DATA, multiclass_=multiclass)
    assert metrics_obj == result


def test_get_metric_objects_warning(get_metric_objects_result: tuple):
    result, multiclass = get_metric_objects_result
    warning_data = GET_METRIC_OBJECTS_DATA.copy()
    warning_data.update({"test": []})

    with pytest.warns(UserWarning, match="WARNING! Custom object 'test', not implemented!"):
        metrics_obj = get_metric_objects(metrics=warning_data, multiclass_=multiclass)
    assert metrics_obj == result


GET_METRIC_DATA = [("multi", True),
                   ("binary", False)]


@pytest.mark.parametrize("typ,multi", GET_METRIC_DATA)
def test_get_metric(metric: dict, typ: str, multi: bool):
    assert get_metric(multiclass_=multi) == metric[typ]


def test_get_model(get_model_result: tuple):
    d3, typ, result = get_model_result
    model = get_model(d3=d3, typ=typ)
    if typ == "Normal":
        for (k_m, v_m), (k_r, v_r) in zip(model.items(), result.items()):
            assert k_m == k_r and v_m == v_r
    else:
        assert model == result
