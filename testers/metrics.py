import numpy as np
from sklearn.metrics import confusion_matrix


class Metrics:
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        return cm

    @staticmethod
    def get_tf_fn_tp_tn(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

    @staticmethod
    def sensitivity_tpfn(FP, FN, TP, TN):
        return TP / (TP + FN)

    @staticmethod
    def sensitivity_y(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        return TP / (TP + FN)

    @staticmethod
    def specificity_tpfn(FP, FN, TP, TN):
        Specificity = TN / (TN + FP)
        return Specificity

    @staticmethod
    def specificity_y(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion matrix')
        print(cm)

        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        Specificity = TN / (TN + FP)
        return Specificity

    @staticmethod
    def mcc_tpfn(FP, FN, TP, TN):
        FP *= 0.0001
        FN *= 0.0001
        TP *= 0.0001
        TN *= 0.0001

        mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        return mcc

    @staticmethod
    def mcc_y(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion matrix')
        print(cm)

        FP = (cm.sum(axis=0) - np.diag(cm)).astype(np.double)
        FN = (cm.sum(axis=1) - np.diag(cm)).astype(np.double)
        TP = (np.diag(cm)).astype(np.double)
        TN = (cm.sum() - (FP + FN + TP)).astype(np.double)

        FP *= 0.0001
        FN *= 0.0001
        TP *= 0.0001
        TN *= 0.0001

        # print('FP', FP)
        # print('FN', FN)
        # print('TP', TP)
        # print('TN', TN)
        # print('up1', TP * TN)
        # print('up2', FP * FN)
        # print('up', (TP * TN) - (FP * FN))
        # print('down1', (TP + FP))
        # print('down2', (TP + FN))
        # print('down3', (TN + FP))
        # print('down4', (TN + FN))
        # print('down', (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
        # print('down sqrt', np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)).astype(int))

        mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        return mcc
