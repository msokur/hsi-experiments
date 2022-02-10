import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, matthews_corrcoef, auc
from tqdm import tqdm
import csv
import datetime

def sensitivity(y_true,y_pred):
    cm=confusion_matrix(y_true, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    Sensitivity = TP/(TP+FN)    
    return Sensitivity

def specificity(y_true,y_pred):
    cm=confusion_matrix(y_true, y_pred)
    print('Confusion matrix')
    print(cm)

    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    Specificity = TN/(TN+FP)    
    return Specificity




with open('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/jupyter/mat_metrics_by_checkpoint.csv', 'a', newline='') as csvfile:
    fieldnames = ['time', 'checkpoint', 'sensitivity_mean', 'specificity_mean', 'sensitivity_median', 'specificity_median']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({'time':'Time',
                     'checkpoint':'Checkpoint',
                     'sensitivity_mean':'sensitivity_mean',
                     'specificity_mean':'specificity_mean', 
                     'sensitivity_median':'sensitivity_median',
                     'specificity_median':'specificity_median'})
    
    for cp in tqdm(np.linspace(42, 60, 10)):
        print('-----------------Checkpoint: '+str(cp)+'------------------- ')
        cp = int(cp)

        gt_s = np.load('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/test/CV_3d_bea_fixed_class_weights/'+f'cp-{cp:04d}'+'/gt_by_patient.npy',
                      allow_pickle=True)
        pred_s = np.load('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/test/CV_3d_bea_fixed_class_weights/'+f'cp-{cp:04d}'+'/predictions_by_patient.npy',
                        allow_pickle=True)

        sensitivities = []
        specificities = []
        f1_scores = []
        aucs = []
        mccs = []

        for i in range(10):
            print('Number', i+1)
            gt = np.array(gt_s[i])
            print('gt', gt.shape, np.unique(gt), gt.dtype)
            pred = np.array(pred_s[i])
            pred = np.argmax(pred, axis=1)
            #print(pred.shape, np.unique(pred), pred.dtype)

            spec = specificity(gt, pred)
            sens = sensitivity(gt, pred)
            f1_score_ = f1_score(gt, pred, average=None)
            auc_ = auc(gt, pred, average = None)
            mcc = matthews_corrcoef(gt, pred, average = None)

            sensitivities.append(sens)
            specificities.append(spec)
            f1_scores.append(f1_score_)
            aucs.append(auc_)
            mccs.append(mcc)

            print('Specificity', spec)
            print('Sensitivity', sens)

            print('------')

        sensitivity_median = np.nanmedian(sensitivities, axis=0)
        specificity_median = np.nanmedian(specificities, axis=0)
        sensitivity_mean = np.nanmean(sensitivities, axis=0)
        specificity_mean = np.nanmean(specificities, axis=0)


        

        writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                         'checkpoint':str(cp),
                         'sensitivity_mean':str(sensitivity_mean),
                         'specificity_mean':str(specificity_mean), 
                         'sensitivity_median':str(sensitivity_median),
                         'specificity_median':str(specificity_median)})