import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.special import softmax
from tqdm import tqdm
import csv
import datetime
import os
import matplotlib.pylab as plt
import abc

import config
from metrics import Metrics


class EvaluationBase(Metrics):
    def __init__(self, test_preffix='C:\\Users\\tkachenko\\Desktop\\HSI\\hsi-experiments\\test'):
        super().__init__()
        self.test_preffix = test_preffix
        self.labels = config.TISSUE_LABELS
        self.labels_of_classes_to_train = config.LABELS_OF_CLASSES_TO_TRAIN
        self.label_color = config.PLOT_COLORS

    @abc.abstractmethod
    def get_csv_fieldnames(self, metrics_dict, metrics_from_scores_dict):
        pass

    @abc.abstractmethod
    def write_metrics_to_csv(self, writer, metrics):
        pass

    @abc.abstractmethod
    def get_fpr_tpr(self, gt, predictions):
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
        print(metrics_)

        self.write_metrics_to_csv(writer, metrics_)

        return metrics_

    def evaluate_checkpoints(self, name, cp_rng, mode='w', print_curve=False, threshold=None):

        analyse_path = os.path.join(self.test_preffix, name, 'analyse')
        if not os.path.exists(analyse_path):
            os.mkdir(analyse_path)

        with open(os.path.join(analyse_path, name + '.csv'), mode, newline='') as csvfile:
            fieldnames = ['time', 'checkpoint', 'sensitivity_mean', 'specificity_mean']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'time': 'Time',
                             'checkpoint': 'Checkpoint',
                             'sensitivity_mean': 'sensitivity_mean',
                             'specificity_mean': 'specificity_mean'})

            # for cp in tqdm(np.linspace(cp_rng[0], cp_rng[1], cp_rng[2])):
            for cp in tqdm(cp_rng):
                print('-----------------Checkpoint: ' + str(cp) + '------------------- ')
                cp = int(cp)

                gt_s, pred_s = self.read_gt_and_predictions(name, cp)

                sensitivities = []
                specificities = []

                metrics_dict = self.get_metrics_dict()
                metrics_from_scores_dict = self.get_metrics_scores_dict()

                with open(os.path.join(analyse_path, name + f'cp-{cp:04d}' + '.csv'), 'w', newline='') as csvfile_cp:
                    # -------fieldnames
                    fieldnames_cp = self.get_csv_fieldnames(metrics_dict, metrics_from_scores_dict)
                    writer_cp = csv.DictWriter(csvfile_cp, fieldnames=fieldnames_cp)
                    writer_cp.writerow(dict(zip(fieldnames_cp, fieldnames_cp)))
                    print('fieldnames_cp', fieldnames_cp)

                    # --------count metrics

                    for patient_index in range(gt_s.shape[0]):
                        print('Patient', patient_index + 1)
                        gt = np.array(gt_s[patient_index])
                        print('gt', gt.shape, np.unique(gt))
                        predictions_raw = np.array(pred_s[patient_index])
                        predictions = self.count_predictions(predictions_raw, threshold)

                        metrics_ = self.save_metrics(gt, predictions, predictions_raw, writer_cp)

                        sensitivities.append(metrics_["Sensitivity"])
                        specificities.append(metrics_["Specificity"])

                        if print_curve:
                            fpr, tpr = self.get_fpr_tpr(gt, predictions_raw)
                            self.print_roc_curve(fpr,
                                                 tpr,
                                                 metrics_["AUC"],
                                                 f'Image_{patient_index}',
                                                 analyse_path)

                sensitivity_mean = np.nanmean(sensitivities, axis=0)
                specificity_mean = np.nanmean(specificities, axis=0)

                writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'checkpoint': str(cp),
                                 'sensitivity_mean': str(sensitivity_mean),
                                 'specificity_mean': str(specificity_mean)})

    def print_roc_curve(self, fpr, tpr, auc, name, save_pht):
        print("Let's draw a plot")
        ax = plt.axes()
        ax.set_facecolor('gainsboro')

        for idx in self.labels_of_classes_to_train:
            plt.plot(fpr[idx], tpr[idx], color=self.label_color[idx], lw=2,
                     label='ROC curve of {0} (area = {1:0.2f})'
                           ''.format(self.labels[idx], auc[idx]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic for multi-class data\n Image: {name}')
        plt.legend(loc="lower right")
        print(os.path.join(save_pht, f'ROC_{name}.png'))
        plt.savefig(os.path.join(save_pht, f'ROC_{name}.png'))
        # plt.show()
        plt.clf()

    def read_gt_and_predictions(self, name, cp):
        gt_s = np.load(os.path.join(self.test_preffix, name, f'cp-{cp:04d}', 'gt_by_patient.npy'),
                       allow_pickle=True)
        pred_s = np.load(
            os.path.join(self.test_preffix, name, f'cp-{cp:04d}', 'predictions_by_patient.npy'),
            allow_pickle=True)

        return gt_s, pred_s


if __name__ == '__main__':
    mat_validator = EvaluationBase()
    mat_validator.validate_checkpoints(config.bea_db, [38], print_curve=True)
