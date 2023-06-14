from util.tf_metric_multiclass import F1_score
import pytest
import numpy as np

y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
y_pred = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0],
          [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1],
          [0, 0, 1]]
micro = 0.6190
macro = 0.5686
weighted = 0.5947
multi = [0.8421, 0.3636, 0.5000]

test_data = [("micro", y_true, y_pred, micro),
             ("macro", y_true, y_pred, macro),
             ("weighted", y_true, y_pred, weighted)]

test_data_multi = [("multi", y_true, y_pred, multi)]

test_data_merge = [("micro", y_true, y_pred, micro),
                   ("macro", y_true, y_pred, macro),
                   ("weighted", y_true, y_pred, weighted)]

test_data_multi_merge = [("multi", y_true, y_pred, multi)]


@pytest.fixture()
def num_classes():
    return 3


@pytest.mark.parametrize("average,y_true_,y_pred_,result", test_data)
def test_f1_score(num_classes, average, y_true_, y_pred_, result):
    f1_s = F1_score(num_classes=num_classes, average=average)
    assert 0.0 == f1_s.result().numpy()

    f1_s.update_state(y_true=y_true_, y_pred=y_pred_)
    assert result == pytest.approx(f1_s.result().numpy(), 0.0001)

    f1_s.reset_state()
    assert 0.0 == f1_s.result().numpy()


@pytest.mark.parametrize("average,y_true_,y_pred_,result", test_data_multi)
def test_multi_f1_score(num_classes, average, y_true_, y_pred_, result):
    f1_s = F1_score(num_classes=num_classes, average=average)
    assert (np.zeros(num_classes, dtype=np.float32) == f1_s.result().numpy()).all()

    f1_s.update_state(y_true=y_true_, y_pred=y_pred_)
    assert (np.array(result, dtype=np.float32) == np.round(f1_s.result().numpy(), 4)).all()

    f1_s.reset_state()
    assert (np.zeros(num_classes, dtype=np.float32) == f1_s.result().numpy()).all()


@pytest.mark.parametrize("average,y_true_,y_pred_,result", test_data_merge)
def test_merge_state_f1_score(num_classes, average, y_true_, y_pred_, result):
    f1_s1 = F1_score(num_classes=num_classes, average=average)
    f1_s1.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s2 = F1_score(num_classes=num_classes, average=average)
    f1_s2.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s3 = F1_score(num_classes=num_classes, average=average)
    f1_s3.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s1.merge_state([f1_s2, f1_s3])

    assert result == pytest.approx(f1_s1.result().numpy(), 0.0001)


@pytest.mark.parametrize("average,y_true_,y_pred_,result", test_data_multi_merge)
def test_multi_merge_state_f1_score(num_classes, average, y_true_, y_pred_, result):
    f1_s1 = F1_score(num_classes=num_classes, average=average)
    f1_s1.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s2 = F1_score(num_classes=num_classes, average=average)
    f1_s2.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s3 = F1_score(num_classes=num_classes, average=average)
    f1_s3.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s1.merge_state([f1_s2, f1_s3])

    assert (np.array(result, dtype=np.float32) == np.round(f1_s1.result().numpy(), 4)).all()


def test_raise_value_error_for_wrong_average(num_classes):
    with pytest.raises(ValueError):
        F1_score(num_classes=num_classes, average=None)

    with pytest.raises(ValueError):
        F1_score(num_classes=num_classes, average="sum")


def test_raise_value_error_merge_state_f1_score(num_classes):
    f1_s1 = F1_score(num_classes=num_classes, average="macro")
    f1_s1.update_state(y_true=y_true, y_pred=y_pred)

    f1_s2 = F1_score(num_classes=num_classes, average="multi")
    f1_s2.update_state(y_true=y_true, y_pred=y_pred)

    with pytest.raises(ValueError):
        f1_s1.merge_state([f1_s2])
