import os
from glob import glob
import abc
import datetime
import numpy as np
import csv
from tqdm import tqdm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import config
import utils
from provider import get_trainer, get_data_loader
from data_utils.data_loaders.data_loader_base import DataLoader
from metrics.predictor import Predictor


class CrossValidatorBase:
    def __init__(self, name):
        # self.metrics_saving_path = config.TEST_NPZ_PATH
        self.name = name

        if config.MODE == 'CLUSTER':
            self.prefix = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments'
        else:
            # self.prefix = 'C:\\Users\\tkachenko\\Desktop\\HSI\\'
            self.prefix = 'C:\\Users\\tkachenko\\Desktop\\HSI\\hsi-experiments'

        self.predictions_filename = 'predictions_by_patient.npy',
        self.gt_filename = 'gt_by_patient.npy'
        self.metrics_for_threshold_filename_base = 'metrics_by_threshold_'

    @staticmethod
    def get_execution_flags():
        return {
            "cross_validation": True,
            "evaluation": True
        }

    def pipeline(self, execution_flags={}, **kwargs):
        if not execution_flags:
            execution_flags = CrossValidatorBase.get_execution_flags()

        if execution_flags['cross_validation']:
            CrossValidatorBase.cross_validation(self.name)
        if execution_flags['evaluation']:
            self.evaluation(**kwargs)

        utils.send_tg_message(
            f'Mariia, operations in cross_validation.py for {self.name} are successfully completed!')

    @abc.abstractmethod
    def evaluation(self, **kwargs):  # has to be implemented in child classes
        pass

    @staticmethod
    def cross_validation_step(model_name, except_names=[]):
        trainer = get_trainer(model_name=model_name, except_indexes=except_names)
        trainer.train()

    @staticmethod
    def cross_validation(root_folder_name, csv_filename=None):
        config.MODEL_NAME_PATHS.append(root_folder_name)

        root_folder = os.path.join(*config.MODEL_NAME_PATHS)
        config.MODEL_NAME = config.get_model_name(config.MODEL_NAME_PATHS)

        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        data_loader = get_data_loader()
        paths, splits = data_loader.get_paths_and_splits()

        date_ = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")

        if csv_filename is None:
            csv_filename = os.path.join(root_folder, root_folder_name + '_stats' + date_ + '.csv')

        for indexes in splits[config.CV_FIRST_SPLIT:]:
            model_name = config.MODEL_NAME  # config.MODEL_NAME
            if len(indexes) > 1:
                for i in indexes:
                    model_name += '_' + str(i)
            else:
                model_name += '_' + str(indexes[0]) + '_' + DataLoader.get_name_easy(np.array(paths)[indexes][0])
                # скопировала на всякий случай с сервера, когда-то тут была ошибка
                # config.model_name += '_' + str(indexes[0]) + '_' + np.array(paths)[indexes][0].split("/")[-1].split(".")[0].split('SpecCube')[0]

            print('model_name', model_name)
            paths_patch = np.array(paths)[indexes]

            CrossValidatorBase.cross_validation_step(model_name,
                                                     except_names=[DataLoader.get_name_easy(p) for p in paths_patch])

            for i, path_ in enumerate(paths_patch):
                sensitivity, specificity = 0, 0
                with open(csv_filename, 'a', newline='') as csvfile:  # for full cross_valid and for separate file
                    fieldnames = ['time', 'index', 'sensitivity', 'specificity', 'name', 'model_name']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                     'index': str(i),
                                     'sensitivity': str(sensitivity),
                                     'specificity': str(specificity),
                                     'name': path_,
                                     # 'model_name': config.MODEL_NAME})
                                     'model_name': model_name})

        return csv_filename

    @staticmethod
    def get_best_checkpoint_from_csv(model_path):
        checkpoints_paths = sorted(glob(os.path.join(model_path, 'checkpoints/*/')))
        best_checkpoint_path = checkpoints_paths[-1]
        return best_checkpoint_path.split(config.SYSTEM_PATHS_DELIMITER)[-2]

    @staticmethod
    def get_nearest_int_delimiter(path):
        checkpoints_paths = glob(os.path.join(path, 'cp-*'))
        checkpoints_paths = sorted(checkpoints_paths)

        return int(checkpoints_paths[0].split(config.SYSTEM_PATHS_DELIMITER)[-1].split('-')[-1])

    @staticmethod
    def get_csv(search_path):
        csv_paths = glob(search_path)
        if len(csv_paths) > 1:
            raise ValueError(search_path + ' has more then one .csv files!')
        if len(csv_paths) == 0:
            raise ValueError('No .csv files were found in ' + search_path)
        csv_path = csv_paths[0]

        return csv_path

    @staticmethod
    def get_history(model_path):
        history_paths = utils.glob_multiple_file_types(model_path, '.*.npy', '*.npy')
        # print(history_paths)
        if len(history_paths) == 0:
            print('Error! No history files were found!')
            # raise ValueError('Error! No history files were found!')
            return {}, model_path
        if len(history_paths) > 1:
            print(f'Error! Too many history.npy files were found in {model_path}!')
            return {}, model_path
            # raise ValueError(f'Error! Too many .npy files were found in {model_path}!')

        history_path = history_paths[0]
        history = np.load(history_path, allow_pickle=True)
        # print(history)
        if len(history.shape) == 0:
            history = history.item()
        return history, history_path

    @staticmethod
    def save_roc_auc_plot(pred, gt, save_path):
        fpr, tpr, threshold = metrics.roc_curve(gt, pred)
        print('threshold', threshold, threshold[np.argmax(tpr - fpr)])
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(os.path.join(save_path, 'roc.png'))
        plt.clf()

    #  be careful if the order of metrics_names would be replaced (see checking of mcc and dices)
    def save_metrics_for_threshold(self, folder_with_predictions,
                                   threshold, metrics_, technical_column_names=['Time', 'Name'],
                                   metrics_column_names=['Accuracy',
                                                         'Sensitivity',
                                                         'Specificity',
                                                         'Dice_0',
                                                         'Dice_1',
                                                         'AUC',
                                                         'Mcc',
                                                         'Best_threshold'],
                                   ):
        def get_time_threshold_name(time, name):
            return {
                technical_column_names[0]: time,
                technical_column_names[1]: name
            }

        def replace_commas(arr):
            return [str(r).replace('.', ',') for r in arr]

        def write_row(val_for_time_column, val_for_name_column, values):
            part1 = get_time_threshold_name(val_for_time_column, val_for_name_column)
            part2 = dict(zip(metrics_column_names, replace_commas(values)))
            part1.update(part2)
            writer.writerow(part1)

        if len(metrics_column_names) != len(metrics_):
            raise ValueError("Error! Number of metrics doesn't correspond to number of names for metrics ("
                             "metrics_names)")

        with open(os.path.join(folder_with_predictions,
                               self.metrics_for_threshold_filename_base + str(threshold) + '.csv'), 'w',
                  newline='') as csvfile:
            # write header

            fieldnames = technical_column_names + metrics_column_names
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(dict(zip(fieldnames, fieldnames)))

            # check mcc and dices for 0 to get real mean - be careful if the order would be replaces

            for i in [3, 4, -2]:
                metrics_[i] = np.array(metrics_[i])
                metrics_[i][metrics_[i] == 0] = float('NaN')

            # loading of names
            data = []
            with open(self.training_csv_file, newline='') as csv_file:
                report_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for row in tqdm(report_reader):
                    data.append(row)
            names = np.array(data)[:, 4]

            metrics_.insert(0, names)

            # save metrics for patients

            for row in zip(*metrics_):
                write_row(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), row[0], row[1:])

            # save mean, std and median

            metrics_.pop(0)

            metrics_mean = [np.nanmean(m) for m in metrics_]
            metrics_median = [np.nanmedian(m) for m in metrics_]
            metrics_std = [np.nanstd(m) for m in metrics_]

            for metr_name, metr in zip(["TOTAL MEAN", "TOTAL STD", "TOTAL MEDIAN"],
                                       [metrics_mean, metrics_std, metrics_median]):
                write_row(metr_name, '-', metr)

    def count_metrics_on_diff_thresholds(self, folder_with_predictions,
                                         threshold_range_params=[0, 1, 21],
                                         threshold_range_plain=None):
        predictions_by_patient = np.load(os.path.join(folder_with_predictions, self.predictions_filename),
                                         allow_pickle=True)
        gt_by_patient = np.load(os.path.join(folder_with_predictions, self.gt_filename), allow_pickle=True)

        sensitivities_mean = []
        specificities_mean = []

        rng = np.round(np.linspace(threshold_range_params[0], threshold_range_params[1], threshold_range_params[2]), 4)
        if threshold_range_plain is not None:
            rng = threshold_range_plain.copy()

        for threshold in tqdm(rng):
            sens = []
            spec = []
            dices = []
            best_thresholds = []
            aucs = []
            mcc_s = []
            acc_s = []
            for patient in range(predictions_by_patient.shape[0]):
                print('unique', np.unique(gt_by_patient[patient]))
                print(gt_by_patient[patient].shape)
                print(predictions_by_patient[patient].shape)
                fpr, tpr, thresholds_p = metrics.roc_curve(np.rint(gt_by_patient[patient]),
                                                           np.array(predictions_by_patient[patient]))
                roc_auc = metrics.auc(fpr, tpr)
                predictions = np.array(np.array(predictions_by_patient[patient]) > threshold).astype(np.uint8)

                threshold_p = thresholds_p[np.argmax(tpr - fpr)]
                gt = np.rint(gt_by_patient[patient])

                sensitivity_p, specificity_p, dice_p = Predictor.count_metrics(gt, predictions)
                mcc_p = metrics.matthews_corrcoef(gt, predictions)
                acc_p = metrics.accuracy_score(gt, predictions)

                sens.append(sensitivity_p)
                spec.append(specificity_p)
                dices.append(dice_p)
                best_thresholds.append(threshold_p)
                aucs.append(roc_auc)
                mcc_s.append(mcc_p)
                acc_s.append(acc_p)

            sensitivity_mean = np.nanmean(sens)
            specificity_mean = np.nanmean(spec)

            sensitivities_mean.append(sensitivity_mean)
            specificities_mean.append(specificity_mean)

            best_thresholds = np.array(best_thresholds)
            best_thresholds[best_thresholds > 1.] = np.nan
            dices = np.array(dices)

            self.save_metrics_for_threshold(folder_with_predictions, threshold,
                                            [acc_s, sens, spec, dices[:, 0], dices[:, 1], aucs, mcc_s, best_thresholds])

            csv_with_all_thresholds = os.path.join(folder_with_predictions, 'metrics_threshold_relation_by_patient.csv')
            with open(csv_with_all_thresholds, 'a', newline='') as csvfile:
                fieldnames = ['time',
                              'threshold',
                              'sensitivity_mean',
                              'specificity_mean',
                              'dice_mean']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold': str(threshold),
                                 'sensitivity_mean': str(sensitivity_mean),
                                 'specificity_mean': str(specificity_mean),
                                 'dice_mean': str(np.nanmean(dices))})

        plt.plot(rng, sensitivities_mean)
        plt.plot(rng, specificities_mean)
        plt.savefig(os.path.join(folder_with_predictions, 'thresholds_metrics_curves_mean.png'))
        plt.clf()

    def save_predictions(self, csv_path,
                         save_folder,
                         checkpoint=None,
                         save_roc_auc_curve=False):
        """

        :param csv_path:
        0 - date
        1 - index
        2 - sensitivity
        3 - specificity
        4 - .dat path
        5 - model path
        """
        self.metrics_saving_folder = save_folder

        if checkpoint is None:
            f'cp-{config.EPOCHS:04d}'

        all_predictions_raw = []
        predictions_by_patient = []
        all_gt = []
        gt_by_patient = []

        with open(csv_path, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                print(', '.join(row))

                # checkpoint_number = int(checkpoint.split('-')[-1])
                # print(checkpoint_number)

                model_path = row[5]
                if 'LOCAL' in config.MODE:
                    model_path = row[5].split('hsi-experiments')[-1][1:]

                checkpoint = CrossValidatorBase.get_best_checkpoint_from_csv(model_path)
                print(f'We get checkpoint {checkpoint} for {model_path}')

                tester = Predictor(checkpoint, ['data'], '', MODEL_FOLDER=model_path)

                name = DataLoader.get_name_easy(row[4], delimiter='/')

                predictions, gt = tester.get_predictions_for_npz(os.path.join(self.test_path, name + ".npz"))

                predictions_by_patient.append(predictions)
                gt_by_patient.append(gt)

                if save_roc_auc_curve:
                    all_predictions_raw += predictions
                    all_gt += gt

        # saving of predictions
        np.save(os.path.join(save_folder, self.predictions_filename), predictions_by_patient)
        np.save(os.path.join(save_folder, self.gt_filename), gt_by_patient)

        # roc auc part
        if save_roc_auc_curve:
            CrossValidatorBase.save_roc_auc_plot(all_predictions_raw, all_gt, save_folder)

    def save_predictions_and_metrics_for_checkpoint(self, checkpoint,
                                                    save_folder_,
                                                    training_csv_path,
                                                    thr_ranges=[],
                                                    execution_flags=[True],
                                                    predictions_filename='predictions_by_patient.npy',
                                                    gt_filename='gt_by_patient.npy'):
        self.predictions_filename = predictions_filename
        self.gt_filename = gt_filename

        if type(checkpoint) == int:
            checkpoint = f'cp-{checkpoint:04d}'

        self.training_csv_file = training_csv_path
        print('CHECKPOINT: ', checkpoint)

        if not os.path.exists(save_folder_):
            os.mkdir(save_folder_)

        save_folder = os.path.join(save_folder_, checkpoint)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        if execution_flags[0]:
            self.save_predictions(training_csv_path, save_folder, checkpoint=checkpoint)

        if len(thr_ranges) == 0:
            print('WARNING! No thresholds were given!')

        for rng in thr_ranges:
            self.count_metrics_on_diff_thresholds(save_folder, threshold_range_params=rng)
