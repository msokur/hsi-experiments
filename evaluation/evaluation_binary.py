from evaluation.evaluation_base import EvaluationBase
import datetime
import numpy as np


class EvaluationBinary(EvaluationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_csv_fieldnames(self, metrics_dict, metrics_from_scores_dict):
        fieldnames = np.array(["Time"] + list(metrics_dict.keys()) + list(metrics_from_scores_dict.keys()),
                              dtype=object)
        dice_index = np.flatnonzero(fieldnames == "F1-score")
        fieldnames[dice_index] = "F1-score_healthy"
        fieldnames = np.insert(fieldnames, dice_index + 1, "F1-score_cancer")
        fieldnames = self.add_additional_column_fieldnames(fieldnames)

        print(fieldnames)

        return fieldnames

    def write_metrics_to_csv(self, writer, metrics, time_string=None):
        csv_row = {}
        dice_index = np.flatnonzero(np.array(list(metrics.keys())) == 'F1-score')[0]

        for metric_name, metric_value in list(metrics.items())[:dice_index] + list(metrics.items())[dice_index + 1:]:
            csv_row[metric_name] = metric_value
        csv_row["F1-score_healthy"] = metrics['F1-score'][0]
        csv_row["F1-score_cancer"] = metrics['F1-score'][1]
        if time_string is None:
            csv_row.update({"Time": datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        else:
            csv_row.update({"Time": str(time_string)})
        self.write_additional_columns(csv_row)
        writer.writerow(csv_row)

    def count_predictions(self, predictions, threshold):
        return np.array(np.array(predictions) > threshold).astype(np.uint8)


if __name__ == '__main__':
    pass
