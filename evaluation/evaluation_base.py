import numpy as np
import csv
import datetime
import os
import matplotlib.pylab as plt
import abc

from configuration.parameter import STORAGE_TYPE, DICT_ORIGINAL_NAME
from evaluation.metrics import Metrics
from evaluation.predictor import Predictor
from data_utils.visualization import VisualizationFromData

from configuration.keys import CrossValidationKeys as CVK, PathKeys as PK
from provider import get_data_storage


class EvaluationBase(Metrics):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.config = config
        self.data_storage = get_data_storage(typ=STORAGE_TYPE)
        self.visualization = VisualizationFromData(config=config,
                                                   data_storage=self.data_storage)

        self.results_folder = self.create_joint_folder(self.config.CONFIG_PATHS[PK.RESULTS_FOLDER],
                                                       self.config.CONFIG_CV[CVK.NAME])
        self.results_folder_with_checkpoint = None

        self.comparison_csvname = "compare_all_thresholds.csv"
        self.metrics_filename_base = "metrics_by_threshold"
        self.predictions_npy_filename = 'predictions.npy'
        self.checkpoint_basename = 'cp-'
        self.additional_columns = {}

    @abc.abstractmethod
    def get_csv_fieldnames(self, metrics_dict, metrics_from_scores_dict):
        pass

    @abc.abstractmethod
    def write_metrics_to_csv(self, writer, metrics, time_string=None):
        pass

    @abc.abstractmethod
    def calculate_predictions(self, predictions, threshold):
        pass

    def save_predictions_and_metrics(self, training_csv_path=None,  # for predictions
                                     data_folder=None,  # for predictions
                                     checkpoints=None,  # for predictions and evaluation
                                     thresholds=None,  # for evaluation of binary classification
                                     predictions_npy_filename=None  # for predictions and evaluation
                                     ):
        """
        Save predictions first (using training_csv_path, npz_folder
        and predictions_npy_filename)
        And then calculate metrics and save them to the appropriate .csv (using checkpoints,
        threshold, save_curves and predictions_npy_filename)

        Args:
            training_csv_path:
                path to .csv file from cross_validation (in logs/cross_validation_name folder)
            data_folder:
                path to the folder where .npz of patients are (usually it is config.RAW_NPZ_PATH)
            checkpoints:
                raw list of checkpoints to test (None, if EarlyStopping was used)
            thresholds:
                raw list of thresholds to test (None, if multiclass)
            predictions_npy_filename:
                By default predictions are saved to predictions.npy in
                metrics/cross_validation_name/cp-checkpoint_number/
                folder. With this parameter it's possible to give it another name. For general metrics calculation is
                not necessary

        Restrictions:
            1. If you want to check some checkpoints manually
            you have to set CONFIG_PATHS['GET_CHECKPOINT_FROM_VALID'] = False
            You don't need to specify checkpoints if you want to use the best checkpoint found by EarlyStopping
            2. If evaluation is multiclass and thresholds are specified ValueError will be raised

        Returns:
            nothing

        """
        if thresholds is not None and not all(t > 0 for t in thresholds):
            raise ValueError(f'Error! Some of thresholds are negative: {thresholds}. Please check.')

        self.calculate_and_save_predictions(checkpoints=checkpoints,
                                            data_folder=data_folder,
                                            training_csv_path=training_csv_path,
                                            predictions_npy_filename=predictions_npy_filename)

        self.evaluate(checkpoints=checkpoints,
                      thresholds=thresholds,
                      save_curves=self.config.CONFIG_CV["SAVE_CURVES"],
                      predictions_npy_filename=predictions_npy_filename)

    def calculate_and_save_predictions(self, checkpoints=None,
                                       data_folder=None,
                                       training_csv_path=None,
                                       predictions_npy_filename=None):
        predictions_npy_filename = self.check_predictions_npy_filename(predictions_npy_filename)
        checkpoints = self.check_and_convert_checkpoints(checkpoints)

        for checkpoint in checkpoints:
            results_folder = self.compose_results_folder_with_checkpoint(checkpoint)

            if data_folder is None:
                data_folder = self.config.CONFIG_PATHS["RAW_NPZ_PATH"]

            Predictor(self.config).save_predictions(training_csv_path,
                                                    data_folder,
                                                    results_folder,
                                                    predictions_npy_filename,
                                                    checkpoint=checkpoint)

    def evaluate(self, checkpoints=None,
                 thresholds=None,
                 save_curves=False,
                 predictions_npy_filename=None):
        def append_value(dict_obj, new_row):
            for key, value in new_row.items():
                if key in dict_obj:
                    if not isinstance(dict_obj[key], list):
                        dict_obj[key] = [dict_obj[key]]
                    dict_obj[key].append(value)
                else:
                    dict_obj[key] = [value]

        predictions_npy_filename = self.check_predictions_npy_filename(predictions_npy_filename)

        checkpoints = self.check_and_convert_checkpoints(checkpoints)
        thresholds = self.check_thresholds(thresholds)

        for checkpoint in checkpoints:
            results_folder = self.compose_results_folder_with_checkpoint(checkpoint)

            for threshold in thresholds:
                comparison_csv_path = os.path.join(results_folder, self.comparison_csvname)

                write_header = True
                if os.path.exists(comparison_csv_path):
                    write_header = False

                with open(comparison_csv_path, "a", newline='') as csvfile:
                    fieldnames = ['Time', 'Checkpoint', 'Threshold', 'Sensitivity_mean', 'Specificity_mean']
                    if self.additional_columns:
                        fieldnames += list(self.additional_columns.keys())
                        # print(fieldnames)
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()

                    print(f'-----------------Checkpoint: {checkpoint}, Threshold: {threshold}------------------- ')

                    data = np.load(os.path.join(results_folder, predictions_npy_filename), allow_pickle=True)

                    metrics_all = {}

                    metrics_dict = self.get_metrics_dict()
                    metrics_from_scores_dict = self.get_metrics_scores_dict()

                    with open(os.path.join(results_folder, self.metrics_filename_base + f'_{threshold}' + '.csv'),
                              'w', newline='') as csvfile_cp:
                        # ------- fieldnames
                        fieldnames_cp = self.get_csv_fieldnames(metrics_dict, metrics_from_scores_dict)
                        writer_cp = csv.DictWriter(csvfile_cp, fieldnames=fieldnames_cp)
                        writer_cp.writeheader()

                        # -------- evaluation

                        for patient in data:
                            name = patient['name']
                            gt = patient['gt']
                            predictions_raw = patient["predictions"]
                            predictions = self.calculate_predictions(predictions_raw, threshold)

                            metrics_ = self.save_metrics(gt, predictions, predictions_raw, writer_cp)
                            append_value(metrics_all, metrics_)

                            if save_curves:
                                names = patient[DICT_ORIGINAL_NAME]
                                roc_folder = os.path.join(results_folder, f"roc_by_threshold_{threshold}")
                                if not os.path.exists(roc_folder):
                                    os.mkdir(roc_folder)
                                self.save_roc_curves(gt,
                                                     predictions_raw,
                                                     f'Image_{name}',
                                                     roc_folder)
                                self.visualization.create_and_save_error_maps(save_path=results_folder,
                                                                              threshold=threshold,
                                                                              y_true=gt,
                                                                              y_pred=predictions,
                                                                              original_names=names,
                                                                              patient_name=name)
                        print(f'metrics all {metrics_all}')
                        self.write_total_metrics(writer_cp, metrics_all)

                    sensitivity_mean = np.nanmean(metrics_all['Sensitivity'], axis=0)
                    specificity_mean = np.nanmean(metrics_all['Specificity'], axis=0)

                    if save_curves:
                        self.plot_sensitivity_specificity(sensitivity_mean, specificity_mean,
                                                          results_folder, threshold=threshold)

                    self.write_row_to_comparison_file(checkpoint, threshold, sensitivity_mean, specificity_mean, writer)

    def check_predictions_npy_filename(self, predictions_npy_filename):
        if predictions_npy_filename is None:
            return self.predictions_npy_filename
        return predictions_npy_filename

    def check_and_convert_checkpoints(self, checkpoints):
        print(f'Specified checkpoints: {checkpoints}')
        if self.config.CONFIG_CV[CVK.GET_CHECKPOINT_FROM_EARLYSTOPPING] and (checkpoints is not None):
            raise ValueError("Error! CONFIG_CV['GET_CHECKPOINT_FROM_EARLYSTOPPING'] is True (it means that the last "
                             "checkpoint will be taken for each patient from EarlyStopping) and checkpoints are "
                             "specified. Please don't specify checkpoints or set "
                             "GET_CHECKPOINT_FROM_EARLYSTOPPING to False")
        if self.config.CONFIG_CV[CVK.GET_CHECKPOINT_FROM_EARLYSTOPPING]:
            return [None]

        return [f'{self.checkpoint_basename}{checkpoint:04d}' for checkpoint in checkpoints]

    def check_thresholds(self, thresholds):
        print(f'Thresholds: {thresholds}')
        if thresholds is not None and len(self.labels_of_classes_to_train) > 2:
            raise ValueError("Error! Thresholds are specified, but classification is multiclass")
        if thresholds is None and len(self.labels_of_classes_to_train) == 2:
            raise ValueError("Error! You have to specify thresholds for evaluation, because classification is binary")

        if len(self.labels_of_classes_to_train) > 2:
            thresholds = [None]
        return thresholds

    def compose_results_folder_with_checkpoint(self, checkpoint):
        # Singleton
        if self.results_folder_with_checkpoint is None:
            if checkpoint is None:
                self.results_folder_with_checkpoint = self.create_joint_folder(self.results_folder,
                                                                               "Results_with_EarlyStopping")
            else:
                self.results_folder_with_checkpoint = self.create_joint_folder(self.results_folder, checkpoint)
        return self.results_folder_with_checkpoint

    def save_metrics(self, gt, predictions, predictions_raw, writer):
        metrics_dict = self.get_metrics_dict()
        metrics_from_scores_dict = self.get_metrics_scores_dict()

        metrics_ = self.calculate_metrics(metrics_dict, gt, predictions)
        metrics_from_scores_ = self.calculate_metrics(metrics_from_scores_dict, gt, predictions_raw)
        metrics_.update(metrics_from_scores_)

        self.write_metrics_to_csv(writer, metrics_)

        return metrics_

    def write_total_metrics(self, writer_cp, metrics_all):
        mean, std, median = {}, {}, {}
        for k, v in metrics_all.items():
            nan_bool = np.isnan(v).all(axis=0)
            if np.any(nan_bool):
                if isinstance(nan_bool, np.bool_):
                    mean[k], std[k], median[k] = float("NaN"), float("NaN"), float("NaN")
                else:
                    mean[k], std[k], median[k] = self.nan_calc_list(value=v, nan_bool_index=nan_bool)
            else:
                mean[k] = np.nanmean(v, axis=0)
                std[k] = np.nanstd(v, axis=0)
                median[k] = np.nanmedian(v, axis=0)

        self.write_metrics_to_csv(writer_cp, mean, time_string="TOTAL MEAN")
        self.write_metrics_to_csv(writer_cp, std, time_string="TOTAL STD")
        self.write_metrics_to_csv(writer_cp, median, time_string="TOTAL MEDIAN")

    @staticmethod
    def nan_calc_list(value, nan_bool_index):
        mean = [np.nanmean(np.array(value)[:, idx], axis=0) if not nan else float("NaN")
                for idx, nan in enumerate(nan_bool_index)]
        std = [np.nanstd(np.array(value)[:, idx], axis=0) if not nan else float("NaN")
               for idx, nan in enumerate(nan_bool_index)]
        median = [np.nanmedian(np.array(value)[:, idx], axis=0) if not nan else float("NaN")
                  for idx, nan in enumerate(nan_bool_index)]

        return mean, std, median

    @staticmethod
    def plot_sensitivity_specificity(sensitivity_mean, specificity_mean, results_folder, threshold):
        plt.plot(sensitivity_mean, "o", label="sensitivity mean")
        plt.plot(specificity_mean, "o", label="specificity mean")
        plt.ylabel("Value")
        plt.xlabel("Labels of Classes")
        plt.legend(loc="lower right")
        plt.title("Sensitivity and specificity mean by classes")
        plt.savefig(os.path.join(results_folder,
                                 f'thresholds_{threshold}_metrics_curves_mean.png'))
        plt.clf()
        plt.cla()
        plt.close('all')

    def write_row_to_comparison_file(self, cp, threshold, sensitivity_mean, specificity_mean, writer):
        results_row = {'Time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                       'Checkpoint': str(cp),
                       'Threshold': str(threshold),
                       'Sensitivity_mean': str(sensitivity_mean),
                       'Specificity_mean': str(specificity_mean)}
        if self.additional_columns:
            results_row.update(self.additional_columns)

        writer.writerow(results_row)

    def write_additional_columns(self, csv_row):
        if self.additional_columns:
            csv_row.update(self.additional_columns)

    def add_additional_column_fieldnames(self, fieldnames):
        if self.additional_columns:
            fieldnames = list(fieldnames)
            fieldnames += list(self.additional_columns.keys())
            fieldnames = np.array(fieldnames)
        return np.array(fieldnames)

    @staticmethod
    def create_joint_folder(*args):
        results_folder = os.path.join(*args)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        return results_folder


if __name__ == '__main__':
    pass
