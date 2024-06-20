import numpy as np
import datetime

from evaluation.evaluation_base import EvaluationBase


class EvaluationMulticlass(EvaluationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_csv_fieldnames(self, metrics_dict, metrics_from_scores_dict):
        labels = np.empty((len(self.labels_of_classes_to_train),
                           len(metrics_dict) - 1 + len(metrics_from_scores_dict)),
                          dtype=object)  # len(metrics_dict) - 1, because without accuracy

        for idx in self.labels_of_classes_to_train:
            label = self.labels_names[idx]
            names = []
            for key in list(metrics_dict.keys())[1:]:  # add classnames for all metrics after accuracy
                names.append(key + '_' + label)
            for key in list(metrics_from_scores_dict.keys()):
                names.append(key + '_' + label)
            labels[idx] = np.array(names)

        fieldnames = ['Time', 'Accuracy']
        fieldnames = self.add_additional_column_fieldnames(fieldnames).astype(object)
        for m in range(len(metrics_dict) - 1 + len(metrics_from_scores_dict)):
            fieldnames = np.insert(fieldnames, -1, labels[:, m])

        return fieldnames

    def write_metrics_to_csv(self, writer, metrics, time_string=None):
        csv_row = {}
        for metric_name, metric_value in list(metrics.items())[1:]:
            for k, v in zip(self.labels_names.values(), metric_value):
                csv_row[metric_name + '_' + k] = v
        if time_string is None:
            csv_row.update({"Time": datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        else:
            csv_row.update({"Time": str(time_string)})
        csv_row.update({"Accuracy": metrics["Accuracy"]})
        self.write_additional_columns(csv_row)
        writer.writerow(csv_row)

    def calculate_predictions(self, predictions, threshold):
        return np.argmax(predictions, axis=1)


if __name__ == '__main__':
    pass
