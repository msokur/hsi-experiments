from sklearn.metrics import auc, confusion_matrix, f1_score, accuracy_score, roc_curve, roc_auc_score, matthews_corrcoef
import numpy as np

import config


class Metrics:
    def __init__(self):
        if len(config.LABELS_OF_CLASSES_TO_TRAIN) == 2:
            self.binary = True
        else:
            self.binary = False

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
    def count_metrics(metrics_dict, gt, predictions):
        metrics = {}
        for name, metric_func in metrics_dict.items():
            metrics[name] = metric_func(gt, predictions)

        return metrics

    @staticmethod
    def get_fp_fn_tp_tn(gt, predictions):
        cm = confusion_matrix(gt, predictions, labels=config.LABELS_OF_CLASSES_TO_TRAIN)

        tp = (np.diag(cm)).astype(np.double)
        fp = (cm.sum(axis=0) - tp).astype(np.double)
        fn = (cm.sum(axis=1) - tp).astype(np.double)
        tn = (cm.sum() - (fp + fn + tp)).astype(np.double)
        return fp, fn, tp, tn

    @staticmethod
    def accuracy(gt, predictions):
        return accuracy_score(gt, predictions)

    @staticmethod
    def print_confusion_matrix(gt, predictions):
        print('Confusion matrix')
        print(confusion_matrix(gt, predictions, labels=config.LABELS_OF_CLASSES_TO_TRAIN))

    @staticmethod
    def f1(gt, predictions):
        f1_result = np.array([float('NaN')] * len(config.LABELS_OF_CLASSES_TO_TRAIN))
        gt_unique = np.unique(gt)

        f1 = f1_score(gt, predictions, average=None, labels=gt_unique)
        f1_result[gt_unique] = f1
        return f1_result

    def sensitivity(self, gt, predictions):  # recall
        fp, fn, tp, tn = self.get_fp_fn_tp_tn(gt, predictions)
        if self.binary:
            return tp / (tp + fn)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                sensitivity = tp / (tp + fn)

            #sensitivity[np.isnan(sensitivity)] = 0.
            return sensitivity

    def auc(self, gt, predictions):
        gt_unique = np.unique(gt)

        if self.binary:
            if gt_unique.shape[0] != 1:
                auc_ = roc_auc_score(gt, predictions)
            else:
                auc_ = float('NaN')
            return auc_
        else:
            number_of_classes = len(config.LABELS_OF_CLASSES_TO_TRAIN)
            auc_result = np.array([float('NaN')] * number_of_classes)
            if len(gt_unique) > 2:
                auc_ = roc_auc_score(np.eye(number_of_classes)[gt],
                                        predictions,
                                        average=None, labels=gt_unique)   # .reshape(-1, 1)
                auc_result[gt_unique] = auc_
            else:
                for uniq in gt_unique:
                    pred = np.argmax(predictions, axis=1)
                    fpr, tpr, thresholds_p = roc_curve(gt, pred, pos_label=uniq)
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

            #specificity[np.isnan(specificity)] = 0.
            return specificity

    def mcc(self, gt, predictions):
        fp, fn, tp, tn = self.get_fp_fn_tp_tn(gt, predictions)

        if self.binary:
            return matthews_corrcoef(gt, predictions)
        else:
            fp *= 0.0001
            fn *= 0.0001
            tp *= 0.0001
            tn *= 0.0001

            with np.errstate(divide='ignore', invalid='ignore'):
                mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

            #mcc[np.isnan(mcc)] = 0.

            return mcc




