from trainers.utils.custom_metrics.multiclass.f1_score import F1_score
import pytest
import numpy as np

Y_TRUE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
Y_PRED = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0],
          [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1],
          [0, 0, 1]]
MICRO = 0.6190
MACRO = 0.5686
WEIGHTED = 0.5947
MULTI = [0.8421, 0.3636, 0.5000]

TEST_DATA = [("micro", Y_TRUE, Y_PRED, MICRO),
             ("macro", Y_TRUE, Y_PRED, MACRO),
             ("weighted", Y_TRUE, Y_PRED, WEIGHTED)]

TEST_DATA_MULTI = [("multi", Y_TRUE, Y_PRED, MULTI)]

TEST_DATA_MERGE = [("micro", Y_TRUE, Y_PRED, MICRO),
                   ("macro", Y_TRUE, Y_PRED, MACRO),
                   ("weighted", Y_TRUE, Y_PRED, WEIGHTED)]

TEST_DATA_MULTI_MERGE = [("multi", Y_TRUE, Y_PRED, MULTI)]


@pytest.fixture()
def num_classes():
    return 3


@pytest.mark.parametrize("average,y_true_,y_pred_,result", TEST_DATA)
def test_f1_score(num_classes, average, y_true_, y_pred_, result):
    f1_s = F1_score(num_classes=num_classes, average=average)
    assert f1_s.result().numpy() == 0.0

    f1_s.update_state(y_true=y_true_, y_pred=y_pred_)
    assert pytest.approx(f1_s.result().numpy(), 0.0001) == result

    f1_s.reset_state()
    assert f1_s.result().numpy() == 0.0


@pytest.mark.parametrize("average,y_true_,y_pred_,result", TEST_DATA_MULTI)
def test_multi_f1_score(num_classes, average, y_true_, y_pred_, result):
    f1_s = F1_score(num_classes=num_classes, average=average)
    assert (f1_s.result().numpy() == np.zeros(num_classes, dtype=np.float32)).all()

    f1_s.update_state(y_true=y_true_, y_pred=y_pred_)
    assert (np.round(f1_s.result().numpy(), 4) == np.array(result, dtype=np.float32)).all()

    f1_s.reset_state()
    assert (f1_s.result().numpy() == np.zeros(num_classes, dtype=np.float32)).all()


@pytest.mark.parametrize("average,y_true_,y_pred_,result", TEST_DATA_MERGE)
def test_merge_state_f1_score(num_classes, average, y_true_, y_pred_, result):
    f1_s1 = F1_score(num_classes=num_classes, average=average)
    f1_s1.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s2 = F1_score(num_classes=num_classes, average=average)
    f1_s2.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s3 = F1_score(num_classes=num_classes, average=average)
    f1_s3.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s1.merge_state([f1_s2, f1_s3])

    assert pytest.approx(f1_s1.result().numpy(), 0.0001) == result


@pytest.mark.parametrize("average,y_true_,y_pred_,result", TEST_DATA_MULTI_MERGE)
def test_multi_merge_state_f1_score(num_classes, average, y_true_, y_pred_, result):
    f1_s1 = F1_score(num_classes=num_classes, average=average)
    f1_s1.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s2 = F1_score(num_classes=num_classes, average=average)
    f1_s2.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s3 = F1_score(num_classes=num_classes, average=average)
    f1_s3.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s1.merge_state([f1_s2, f1_s3])

    assert (np.round(f1_s1.result().numpy(), 4) == np.array(result, dtype=np.float32)).all()


def test_raise_value_error_for_wrong_average(num_classes):
    with pytest.raises(ValueError):
        F1_score(num_classes=num_classes, average=None)

    with pytest.raises(ValueError):
        F1_score(num_classes=num_classes, average="sum")


def test_raise_value_error_merge_state_f1_score(num_classes):
    f1_s1 = F1_score(num_classes=num_classes, average="macro")
    f1_s1.update_state(y_true=Y_TRUE, y_pred=Y_PRED)

    f1_s2 = F1_score(num_classes=num_classes, average="multi")
    f1_s2.update_state(y_true=Y_TRUE, y_pred=Y_PRED)

    with pytest.raises(ValueError):
        f1_s1.merge_state([f1_s2])
