from evaluation_base import EvaluationBase
import datetime
import numpy as np
from sklearn.metrics import roc_curve

import config


class EvaluationBinary(EvaluationBase):
    def __init__(self):
        super().__init__()

    def get_csv_fieldnames(self, metrics_dict, metrics_from_scores_dict):
        fieldnames = np.array(list(metrics_dict.keys()) + list(metrics_from_scores_dict.keys()), dtype=object)
        dice_index = np.flatnonzero(fieldnames == "F1-score")
        fieldnames[dice_index] = "F1-score_healthy"
        fieldnames = np.insert(fieldnames, dice_index+1, "F1-score_cancer")
        print(fieldnames)

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
        fpr, tpr, _ = roc_curve(gt, predictions)
        return fpr, tpr

    def count_predictions(self, predictions, threshold):
        return np.array(np.array(predictions) > threshold).astype(np.uint8)

if __name__ == '__main__':
    eval_binary = EvaluationBinary()
    eval_binary.evaluate_checkpoints(config.bea_db, [38], print_curve=True, threshold=0.5)
