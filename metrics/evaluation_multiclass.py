import numpy as np
from sklearn.metrics import roc_curve
import datetime

import config
from evaluation_base import EvaluationBase


class EvaluationMulticlass(EvaluationBase):
    def __init__(self):
        super().__init__()

    def get_csv_fieldnames(self, metrics_dict, metrics_from_scores_dict):
        labels = np.empty((len(config.LABELS_OF_CLASSES_TO_TRAIN),
                           len(metrics_dict) - 1 + len(metrics_from_scores_dict)),
                          dtype=object)  # len(metrics_dict) - 1, because without accuracy

        for idx in self.labels_of_classes_to_train:
            label = self.labels[idx]
            names = []
            for key in list(metrics_dict.keys())[1:]:  # all metrics after accuracy
                names.append(key + '_' + label)
            for key in list(metrics_from_scores_dict.keys()):
                names.append(key + '_' + label)
            labels[idx] = np.array(names)

        print('labels', labels)

        fieldnames = ['Time', 'Accuracy']
        for m in range(len(metrics_dict) - 1 + len(metrics_from_scores_dict)):
            fieldnames.extend(labels[:, m])

        return fieldnames

    def write_metrics_to_csv(self, writer, metrics):
        csv_row = {}
        for metric_name, metric_value in list(metrics.items())[1:]:
            for k, v in zip(config.TISSUE_LABELS.values(), metric_value):
                csv_row[metric_name + '_' + k] = v
        print(csv_row)
        csv_row.update({"Time": datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        csv_row.update({"Accuracy": metrics["Accuracy"]})
        writer.writerow(csv_row)

    def get_fpr_tpr(self, gt, predictions):
        fpr, tpr = [], []
        for idx in np.arange(predictions.shape[-1]):
            fpr_, tpr_, thresholds = roc_curve(gt, predictions[:, idx], pos_label=idx)
            fpr.append(fpr_)
            tpr.append(tpr_)

        return fpr, tpr

    def count_predictions(self, predictions, threshold):
        return np.argmax(predictions, axis=1)


if __name__ == '__main__':
    eval_multiclass = EvaluationMulticlass()
    eval_multiclass.evaluate_checkpoints(config.bea_db, [38], print_curve=True)
