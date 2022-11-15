from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.keras import backend as K


class F1_score(Metric):
    def __init__(self, num_classes, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.f1 = self.add_weight('f1', initializer='zeros')
        self.precision = Precision()
        self.recall = Recall()
        self.range_classes = range(num_classes)

    # @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        f1_scores = []
        # get indexes with the highest value from y_pred
        pred_max = K.argmax(y_pred, axis=1)

        for class_ in self.range_classes:
            # set all indexes true when there the same as the class
            pred_bool = K.equal(pred_max, class_)
            true_bool = K.equal(y_true, class_)

            # convert boolean to float True = 1.0, False = 0.0
            pred_float = K.cast(pred_bool, 'float32')
            true_float = K.cast(true_bool, 'float32')

            p = self.precision(true_float, pred_float)
            r = self.recall(true_float, pred_float)
            f1_score_class = 2 * ((p * r) / (p + r + K.epsilon()))
            f1_scores.append(f1_score_class)
            self.__reset_var()

        # count f1 scores with the value zero
        zero_scores = K.sum(K.cast(K.equal(f1_scores, 0.), 'float32'))
        self.f1.assign(K.sum(f1_scores) / (self.num_classes - zero_scores))

    def result(self):
        return self.f1

    def reset_state(self):
        self.f1.assign(0)
        self.__reset_var()

    def __reset_var(self):
        self.precision.reset_states()
        self.recall.reset_states()


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import f1_score
    from tensorflow_addons.metrics import F1Score

    y_true_ = np.random.randint(7, size=200)
    # y_pred_one = np.reshape(np.random.randint(8, size=200), newshape=(200, 1))
    # np.put_along_axis(y_pred_, y_pred_one, 1, axis=1)
    y_pred_ = np.random.randint(-50.0, 50, (200, 8))
    y_pred_max_ = np.reshape(K.argmax(y_pred_, axis=1), newshape=(200, 1))
    y_pred_2 = np.zeros((200, 8))
    np.put_along_axis(y_pred_2, y_pred_max_, 1, axis=1)
    y_true_2 = np.zeros((200, 8))
    np.put_along_axis(y_true_2, np.reshape(y_true_, newshape=(200, 1)), 1, axis=1)
    f1_score_self = F1_score(8)
    f1_score_self.update_state(y_true_, y_pred_)
    print(f1_score_self.result())
    f1 = f1_score(y_true_2, y_pred_2, average=None)
    print(f1)
    print(np.sum(f1) / 8)

    f1_score_self.update_state(y_true_, y_pred_)
    print(f1_score_self.result())
