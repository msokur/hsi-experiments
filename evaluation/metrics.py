from sklearn.metrics import auc, confusion_matrix, f1_score, accuracy_score, roc_curve, roc_auc_score, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt
import os

from configuration.keys import DataLoaderKeys as DLK


class Metrics:
    def __init__(self, config):
        self.config = config
        self.fill_labels()
        
        if len(self.labels_of_classes_to_train) == 2:
            print('Metrics binary')
            self.binary = True
        else:
            print('Metrics not binary')
            self.binary = False

    def fill_labels(self):
        self.label_color = self.config.CONFIG_DATALOADER[DLK.PLOT_COLORS]
        self.labels_names = self.config.CONFIG_DATALOADER[DLK.TISSUE_LABELS]
        self.labels_of_classes_to_train = self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]
            
    def get_metrics_dict(self):
        return {
            'Accuracy': Metrics.accuracy,
            'Sensitivity': self.sensitivity,
            'Specificity': self.specificity,
            'F1-score': self.f1,
            'MCC': self.mcc
        }

    def get_metrics_scores_dict(self):
        return {
            'AUC': self.auc,
        }

    @staticmethod
    def calculate_metrics(metrics_dict, gt, predictions):
        metrics = {}
        for name, metric_func in metrics_dict.items():
            metrics[name] = metric_func(gt, predictions)

        return metrics

    @staticmethod
    def accuracy(gt, predictions):
        return accuracy_score(gt, predictions)

    def print_confusion_matrix(self, gt, predictions):
        print('Confusion matrix')
        print(confusion_matrix(gt, predictions, labels=self.labels_of_classes_to_train))

    def get_fpr_tpr(self, gt, predictions):
        gt, predictions = np.array(gt), np.array(predictions)
        if self.binary:
            fpr, tpr, _ = roc_curve(gt, predictions)
            return fpr, tpr
        else:
            fpr, tpr = [], []
            gt_unique = np.unique(gt)
            for idx in np.arange(predictions.shape[-1]):
                if idx in gt_unique and gt_unique.shape[0] > 1:
                    fpr_, tpr_, thresholds = roc_curve(gt, predictions[:, idx], pos_label=idx)
                else:
                    fpr_, tpr_ = np.array("NaN"), np.array("NaN")
                fpr.append(fpr_)
                tpr.append(tpr_)

            return np.array(fpr, dtype=object), np.array(tpr, dtype=object)

    def f1(self, gt, predictions):
        f1_result = np.array([float('NaN')] * len(self.labels_of_classes_to_train))
        gt_unique = np.unique(gt)

        f1 = f1_score(gt, predictions, average=None, labels=gt_unique)
        f1_result[gt_unique] = f1
        return f1_result

    def get_fp_fn_tp_tn(self, gt, predictions):
        cm = confusion_matrix(gt, predictions, labels=self.labels_of_classes_to_train)

        if self.binary:
            tn = cm[0, 0]
            tp = cm[1, 1]
            fn = cm[1, 0]
            fp = cm[0, 1]
        else:
            tp = (np.diag(cm)).astype(np.double)
            fp = (cm.sum(axis=0) - tp).astype(np.double)
            fn = (cm.sum(axis=1) - tp).astype(np.double)
            tn = (cm.sum() - (fp + fn + tp)).astype(np.double)
        return fp, fn, tp, tn

    def sensitivity(self, gt, predictions):  # recall
        fp, fn, tp, tn = self.get_fp_fn_tp_tn(gt, predictions)
        if self.binary:
            return tp / (tp + fn)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                sensitivity = tp / (tp + fn)

            # sensitivity[np.isnan(sensitivity)] = 0.
            return sensitivity

    def auc(self, gt, predictions):
        gt_unique = np.unique(gt)

        if self.binary:
            if gt_unique.shape[0] > 1:
                auc_result = roc_auc_score(gt, predictions)
            else:
                auc_result = float('NaN')
        else:
            number_of_classes = len(self.labels_of_classes_to_train)
            if len(gt_unique) == len(self.labels_of_classes_to_train):
                auc_result = roc_auc_score(np.eye(number_of_classes)[gt],
                                           predictions,
                                           average=None, labels=gt_unique)  # .reshape(-1, 1)
            else:
                auc_result = np.array([float('NaN')] * number_of_classes)
                for uniq in gt_unique:
                    if gt_unique.shape[0] < 2:
                        continue
                    predictions_ = np.argmax(predictions, axis=1)
                    fpr, tpr, thresholds_p = roc_curve(gt, predictions_, pos_label=uniq)
                    auc_ = auc(fpr, tpr)
                    auc_result[uniq] = auc_
        return auc_result

    def specificity(self, gt, predictions):
        fp, fn, tp, tn = self.get_fp_fn_tp_tn(gt, predictions)
        if self.binary:
            return tn / (tn + fp)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                specificity = tn / (tn + fp)

            # specificity[np.isnan(specificity)] = 0.
            return specificity

    def mcc(self, gt, predictions):
        fp, fn, tp, tn = self.get_fp_fn_tp_tn(gt, predictions)

        if self.binary:
            if len(np.unique(gt)) == 1:
                return float('NaN')
            return matthews_corrcoef(gt, predictions)
        else:
            fp *= 0.0001
            fn *= 0.0001
            tp *= 0.0001
            tn *= 0.0001

            with np.errstate(divide='ignore', invalid='ignore'):
                mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

            # mcc[np.isnan(mcc)] = 0.

            return mcc

    def save_roc_curves(self, gt, predictions_raw, name, save_folder):
        fpr, tpr = self.get_fpr_tpr(gt, predictions_raw)
        auc_ = self.auc(gt, predictions_raw)

        if len(tpr) > 1:
            for idx in self.labels_of_classes_to_train:
                plt.plot(fpr[idx], tpr[idx], color=self.label_color[idx], lw=2,
                         label='ROC curve of {0} (area = {1:0.2f})'
                               ''.format(self.labels_names[idx], auc_[idx]))
        else:
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {auc_})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic for multi-class data\n Image: {name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_folder, f'ROC_{name}.png'))
        plt.clf()
