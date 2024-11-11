from trainers.utils.custom_metrics.binary.f1_score import F1_score
import pytest
import tensorflow as tf

y_true = tf.Variable([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
y_pred = tf.Variable([1.0, 1.1, 2.2, 1.5, 1.8, 0.9, 1.1, 1.4, 0.5, 0.8])
result10 = 0.7692
result12 = 0.6

test_data = [(y_true, y_pred, 1.0, result10),
             (y_true, y_pred, 1.2, result12)]


@pytest.mark.parametrize("y_true_,y_pred_, threshold, result", test_data)
def test_f1_score(y_true_, y_pred_, threshold, result):
    f1_s = F1_score(threshold=threshold)
    assert f1_s.result().numpy() == 0.0

    f1_s.update_state(y_true=y_true_, y_pred=y_pred_)
    assert pytest.approx(f1_s.result().numpy(), 0.0001) == result

    f1_s.reset_state()
    assert f1_s.result().numpy() == 0.0


@pytest.mark.parametrize("y_true_,y_pred_, threshold, result", test_data)
def test_merge_state_f1_score(y_true_, y_pred_, threshold, result):
    f1_s1 = F1_score(threshold=threshold)
    f1_s1.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s2 = F1_score(threshold=threshold)
    f1_s2.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s3 = F1_score(threshold=threshold)
    f1_s3.update_state(y_true=y_true_, y_pred=y_pred_)

    f1_s1.merge_state([f1_s2, f1_s3])

    assert pytest.approx(f1_s1.result().numpy(), 0.0001) == result
