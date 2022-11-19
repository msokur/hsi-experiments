from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K
import tensorflow as tf


class F1_score(Metric):
    def __init__(self, num_classes, name='f1_score', average='weighted', class_weights=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.f1 = None
        self.average_methode = None
        self.class_weights = None
        self.init_average(self.average, class_weights)

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_max = K.argmax(y_pred, axis=1)
        cm = tf.math.confusion_matrix(y_true, y_pred_max, dtype=tf.float32, num_classes=self.num_classes)
        tp = tf.linalg.diag_part(cm)
        fp = tf.math.reduce_sum(cm, 0) - tp
        fn = tf.math.reduce_sum(cm, 1) - tp
        self.average_methode(tp, fp, fn)

    def macro(self, tp, fp, fn):
        f1_s = tp / (tp + 0.5 * (fp + fn) + K.epsilon())
        self.f1.assign(K.sum(f1_s) / self.num_classes)

    def micro(self, tp, fp, fn):
        tp_sum = K.sum(tp)
        fp_sum = K.sum(fp)
        fn_sum = K.sum(fn)
        f1_s = tp_sum / (tp_sum + 0.5 * (fp_sum + fn_sum) + K.epsilon())
        self.f1.assign(f1_s)

    def weighted(self, tp, fp, fn):
        f1_s = tp / (tp + 0.5 * (fp + fn) + K.epsilon())
        f1_s *= self.class_weights
        self.f1.assign(K.sum(f1_s))

    def multi(self, tp, fp, fn):
        f1_s = tp / (tp + 0.5 * (fp + fn) + K.epsilon())
        self.f1.assign(f1_s)

    def init_average(self, average, class_weights=None):
        if average != self.average:
            self.average = average

        if self.average == 'macro':
            self.average_methode = self.macro
        elif self.average == 'micro':
            self.average_methode = self.micro
        elif self.average == 'weighted':
            self.average_methode = self.weighted
        elif self.average == 'multi':
            self.average_methode = self.multi
            self.f1 = self.add_weight('f1', shape=(self.num_classes,), initializer='zeros')
            if class_weights is not None:
                self.class_weights = [class_weights[key] for key in class_weights.keys()]
            else:
                self.class_weights = [1.0 for weight in range(self.num_classes)]
                print('No class weights are given, so all weights have the value 1.0!')
            return
        else:
            raise ValueError("Only the average Keywords: 'macro', 'micro', 'weighted' and 'multi' are allowed!")

        self.f1 = self.add_weight('f1', initializer='zeros')

    def result(self):
        return self.f1

    def reset_state(self):
        self.f1.assign(tf.zeros(shape=self.f1.shape))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'num_classes': self.num_classes, 'average': self.average,
                'class_weights': self.class_weights}


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import f1_score
    from tensorflow_addons.metrics import F1Score

    y_true_ = np.random.randint(8, size=200)
    # y_pred_one = np.reshape(np.random.randint(8, size=200), newshape=(200, 1))
    # np.put_along_axis(y_pred_, y_pred_one, 1, axis=1)
    y_pred_ = np.random.randint(-50.0, 50, (200, 8))
    maxi = K.argmax(y_pred_, axis=1)
    y_pred_max_ = np.reshape(K.argmax(y_pred_, axis=1), newshape=(200, 1))
    y_pred_2 = np.zeros((200, 8))
    np.put_along_axis(y_pred_2, y_pred_max_, 1, axis=1)
    y_true_2 = np.zeros((200, 8))
    np.put_along_axis(y_true_2, np.reshape(y_true_, newshape=(200, 1)), 1, axis=1)
    f1_score_self = F1_score(8)
    f1_score_self.update_state(y_true_, y_pred_)
    print(f1_score_self.result())
    f1_score_self.reset_state()
    f1 = f1_score(y_true_, y_pred_max_, average=None)
    print(f1)
    print(np.sum(f1) / 8)

    f1_score_self.init_average('multi')
    f1_score_self.update_state(y_true_, y_pred_)
    print(f1_score_self.result())
    f1_score_self.reset_state()

    f1_score_self.init_average('micro')
    f1_score_self.update_state(y_true_, y_pred_)
    print(f1_score_self.result())
    f1_score_self.reset_state()

    cm_ = tf.math.confusion_matrix(y_true_, y_pred_max_, dtype=tf.float32, num_classes=8)
    tp_ = tf.linalg.diag_part(cm_)
    fp_ = tf.math.reduce_sum(cm_, 0) - tp_
    fn_ = tf.math.reduce_sum(cm_, 1) - tp_
    print(f'tp: {tp_}')
    print(f'fp: {fp_}')
    print(f'fn: {fn_}')
