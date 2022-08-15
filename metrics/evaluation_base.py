import numpy as np
from tqdm import tqdm
import csv
import datetime
import os
import inspect
import matplotlib.pylab as plt
import abc

import config
from metrics import Metrics
from predictor import Predictor


class EvaluationBase(Metrics):

    def __init__(self, name, npz_folder=config.TEST_NPZ_PATH, *args, **kwargs):
        super().__init__(*args, **kwargs)

        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parent_dir = os.path.dirname(current_dir)
        self.test_folder = os.path.join(parent_dir, 'test')

        self.save_evaluation_folder = os.path.join(self.test_folder, name)
        if not os.path.exists(self.save_evaluation_folder):
            os.mkdir(self.save_evaluation_folder)

        self.comparable_characteristics_csvname = "compare_all_thresholds.csv"
        self.metrics_filename_base = "metrics_by_threshold"
        self.predictions_npy_filename = 'predictions.npy'

        self.name = name
        self.npz_folder = npz_folder
        self.labels = config.TISSUE_LABELS
        self.label_color = config.PLOT_COLORS

    @abc.abstractmethod
    def get_csv_fieldnames(self, metrics_dict, metrics_from_scores_dict):
        pass

    @abc.abstractmethod
    def write_metrics_to_csv(self, writer, metrics, time_string=None):
        pass

    @abc.abstractmethod
    def count_predictions(self, predictions, threshold):
        pass

    def save_metrics(self, gt, predictions, predictions_raw, writer):
        metrics_dict = self.get_metrics_dict()
        metrics_from_scores_dict = self.get_metrics_scores_dict()

        # --------count metrics
        metrics_ = self.count_metrics(metrics_dict, gt, predictions)
        metrics_from_scores_ = self.count_metrics(metrics_from_scores_dict, gt, predictions_raw)
        metrics_.update(metrics_from_scores_)

        self.write_metrics_to_csv(writer, metrics_)

        return metrics_

    @staticmethod
    def check_checkpoints_for_evaluation(checkpoints_rng, checkpoints_raw_list):
        print(f'checkpoints_rng: {checkpoints_rng}, checkpoints_raw_list: {checkpoints_raw_list}')
        if config.CV_GET_CHECKPOINT_FROM_VALID and (checkpoints_rng is not None or checkpoints_raw_list is not None):
            raise ValueError("Error! config.CV_GET_CHECKPOINT_FROM_VALID is True, it means that the last checkpoint "
                             "will be taken for each patient, and checkpoints_rng (or checkpoints_raw_list) is "
                             "specified at the same time. Please don't specify them or set "
                             "CV_GET_CHECKPOINT_FROM_VALID to False")
        if config.CV_GET_CHECKPOINT_FROM_VALID:
            return [0]

        if checkpoints_rng is None and checkpoints_raw_list is None:
            raise ValueError("Error! Both checkpoints_rng and checkpoints_raw_list are None. Please specify one "
                             "of them")
        if checkpoints_rng is not None and checkpoints_raw_list is not None:
            raise ValueError("Error! Both checkpoints_rng and checkpoints_raw_list are specified. Please specify only "
                             "one of them")

        checkpoints = []
        if checkpoints_raw_list is not None:
            checkpoints = checkpoints_raw_list.copy()
        else:
            checkpoints = np.linspace([checkpoints_rng[0], checkpoints_rng[1], checkpoints_rng[2]])

        return checkpoints

    def check_thresholds_for_evaluation(self, thresholds_range, thresholds_raw_list):
        print(f'thresholds_range: {thresholds_range}, thresholds_raw_list: {thresholds_raw_list}')
        if thresholds_range is not None and thresholds_raw_list is not None:
            raise ValueError("Error! Both thresholds_range and thresholds_raw_list are specified. Please specify only"
                             "one of them")
        if (thresholds_range is not None or thresholds_raw_list is not None) and len(
                self.labels_of_classes_to_train) > 2:
            raise ValueError("Error! Thresholds are specified, but classification is multiclass")

        # sometimes we don't need to count metrics, in this case we just don't specify thresholds
        if thresholds_range is None and thresholds_raw_list is None and len(self.labels_of_classes_to_train) == 2:
            return []

        thresholds = []
        if len(self.labels_of_classes_to_train) > 2:
            thresholds = [None]
        else:
            if thresholds_raw_list is not None:
                thresholds = thresholds_raw_list.copy()
            else:
                thresholds = np.round(np.linspace(thresholds_range[0], thresholds_range[1], thresholds_range[2]), 4)
        return thresholds

    def evaluate(self, checkpoints_rng=None,
                 checkpoints_raw_list=None,
                 thresholds_range=None,
                 thresholds_raw_list=None,
                 save_curves=False):
        def append_value(dict_obj, new_row):
            for key, value in new_row.items():
                if key in dict_obj:
                    if not isinstance(dict_obj[key], list):
                        dict_obj[key] = [dict_obj[key]]
                    dict_obj[key].append(value)
                else:
                    dict_obj[key] = value

        checkpoints = EvaluationBase.check_checkpoints_for_evaluation(checkpoints_rng, checkpoints_raw_list)
        thresholds = self.check_thresholds_for_evaluation(thresholds_range, thresholds_raw_list)

        for cp in checkpoints:
            save_evaluation_folder_with_checkpoint = os.path.join(self.save_evaluation_folder, f'cp-{cp:04d}')

            for threshold in tqdm(thresholds):
                comparable_characteristics_csv_path = os.path.join(save_evaluation_folder_with_checkpoint,
                                                                   self.comparable_characteristics_csvname)
                compare_file_exists = os.path.exists(comparable_characteristics_csv_path)

                with open(comparable_characteristics_csv_path, "a", newline='') as csvfile:
                    fieldnames = ['Time', 'Checkpoint', 'Threshold', 'Sensitivity_mean', 'Specificity_mean']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if not compare_file_exists:
                        writer.writeheader()

                    print(f'-----------------Checkpoint: {cp}, Threshold: {threshold}------------------- ')
                    cp = int(cp)

                    data = np.load(
                        os.path.join(self.save_evaluation_folder, f'cp-{cp:04d}', self.predictions_npy_filename),
                        allow_pickle=True)

                    metrics_all = {}

                    metrics_dict = self.get_metrics_dict()
                    metrics_from_scores_dict = self.get_metrics_scores_dict()

                    with open(os.path.join(save_evaluation_folder_with_checkpoint,
                                           self.metrics_filename_base + f'_{threshold}' + '.csv'),
                              'w', newline='') as csvfile_cp:
                        # -------fieldnames
                        fieldnames_cp = self.get_csv_fieldnames(metrics_dict, metrics_from_scores_dict)
                        writer_cp = csv.DictWriter(csvfile_cp, fieldnames=fieldnames_cp)
                        writer_cp.writeheader()

                        # --------count metrics

                        for patient in data:
                            name = patient['name']
                            gt = np.array(patient['gt'])
                            predictions_raw = np.array(patient["predictions"])
                            predictions = self.count_predictions(predictions_raw, threshold)

                            metrics_ = self.save_metrics(gt, predictions, predictions_raw, writer_cp)

                            append_value(metrics_all, metrics_)

                            if save_curves:
                                self.save_roc_curves(gt,
                                                     predictions_raw,
                                                     f'Image_{name}',
                                                     save_evaluation_folder_with_checkpoint)

                        self.write_metrics_to_csv(writer_cp, {k: np.nanmean(v, axis=0) for k, v in metrics_all.items()},
                                                  time_string="TOTAL MEAN")
                        self.write_metrics_to_csv(writer_cp, {k: np.nanstd(v, axis=0) for k, v in metrics_all.items()},
                                                  time_string="TOTAL STD")
                        self.write_metrics_to_csv(writer_cp,
                                                  {k: np.nanmedian(v, axis=0) for k, v in metrics_all.items()},
                                                  time_string="TOTAL MEDIAN")

                    sensitivity_mean = np.nanmean(metrics_all['Sensitivity'], axis=0)
                    specificity_mean = np.nanmean(metrics_all['Specificity'], axis=0)

                    plt.plot(sensitivity_mean)
                    plt.plot(specificity_mean)
                    plt.savefig(os.path.join(save_evaluation_folder_with_checkpoint,
                                             'thresholds_metrics_curves_mean.png'))

                    writer.writerow({'Time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                     'Checkpoint': str(cp),
                                     'Threshold': str(threshold),
                                     'Sensitivity_mean': str(sensitivity_mean),
                                     'Specificity_mean': str(specificity_mean)})

    def save_predictions_and_metrics(self, training_csv_path=None,   # for predictions
                                     npz_folder=None,             # for predictions
                                     checkpoints_rng=None,            # for predictions and evaluation
                                     checkpoints_raw_list=None,      # for predictions and evaluation
                                     thresholds_range=None,              # for evaluation of binary classification
                                     thresholds_raw_list=None,           # for evaluation of binary classification
                                     save_predictions=False,            # for predictions
                                     save_curves=False):               # for evaluation

        if save_predictions:
            checkpoints = EvaluationBase.check_checkpoints_for_evaluation(checkpoints_rng, checkpoints_raw_list)
            for checkpoint in checkpoints:
                if type(checkpoint) == int:
                    checkpoint = f'cp-{checkpoint:04d}'

                save_folder_with_checkpoint = os.path.join(self.save_evaluation_folder, checkpoint)
                if not os.path.exists(save_folder_with_checkpoint):
                    os.mkdir(save_folder_with_checkpoint)

                if npz_folder is None:
                    npz_folder = self.npz_folder

                Predictor.save_predictions(training_csv_path,
                                           npz_folder,
                                           save_folder_with_checkpoint,
                                           self.predictions_npy_filename,
                                           checkpoint=checkpoint,
                                           save_roc_auc_curve=save_curves)

        self.evaluate(checkpoints_rng=checkpoints_rng,
                      checkpoints_raw_list=checkpoints_raw_list,
                      thresholds_range=thresholds_range,
                      thresholds_raw_list=thresholds_raw_list,
                      save_curves=save_curves)


if __name__ == '__main__':
    mat_validator = EvaluationBase(config.bea_db)
    mat_validator.validate_checkpoints(checkpoints_raw_list=[38], print_curve=False)
