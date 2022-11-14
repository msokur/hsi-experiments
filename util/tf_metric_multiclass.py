from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.keras import backend as K


class F1_score(Metric):
    def __init__(self, num_classes, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.f1 = self.add_weight('f1', initializer='zeros')
        self.precision = Precision()
        self.recall = Recall()

    # @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        f1_scores = []
        classes = self.num_classes
        # get indexes with the highest value from y_pred
        pred_max = K.argmax(y_pred, axis=1)

        for class_ in range(self.num_classes):
            # set all indexes true when there the same as the class
            pred_bool = K.equal(pred_max, class_)
            true_bool = K.equal(y_true, class_)

            # convert boolean to float True = 1.0, False = 0.0
            pred_float = K.cast(pred_bool, 'float32')
            true_float = K.cast(true_bool, 'float32')

            p = self.precision(true_float, pred_float)
            r = self.recall(true_float, pred_float)
            f1_score_class = 2 * ((p * r) / (p + r + K.epsilon()))
            if f1_score_class <= 0:
                classes -= 1
                print(f' Class {class_} has a F1 Score from 0!')
            f1_scores.append(f1_score_class)
            self.__reset_var()

        self.f1.assign(K.sum(f1_scores) / classes)

    def result(self):
        return self.f1

    def reset_states(self):
        self.f1.assign(0)
        self.__reset_var()

    def __reset_var(self):
        self.precision.reset_states()
        self.recall.reset_states()
