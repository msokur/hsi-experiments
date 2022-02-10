import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, matthews_corrcoef, auc, roc_auc_score, accuracy_score
from tqdm import tqdm
import csv
import datetime
import os

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

def mcc(y_true,y_pred):
    cm=confusion_matrix(y_true, y_pred)
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

    #print('FP', FP)
    #print('FN', FN)
    #print('TP', TP)
    #print('TN', TN)
    #print('up1', TP * TN)
    #print('up2', FP * FN)
    #print('up', (TP * TN) - (FP * FN))
    #print('down1', (TP + FP))
    #print('down2', (TP + FN))
    #print('down3', (TN + FP))
    #print('down4', (TN + FN))
    #print('down', (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    #print('down sqrt', np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)).astype(int))

    mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    print('mcc', mcc)
    return mcc


class MatValidator():
    def __init__(self, test_preffix = 'C:\\Users\\tkachenko\\Desktop\\HSI\\hsi-experiments\\test'):
        self.test_preffix = test_preffix

    def validate_checkpoints_brain(self, name, cp_rng, mode='w'):

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

                gt_s = np.load(os.path.join(self.test_preffix, name, f'cp-{cp:04d}', 'gt_by_patient.npy'),
                               allow_pickle=True)
                pred_s = np.load(
                    os.path.join(self.test_preffix, name, f'cp-{cp:04d}', 'predictions_by_patient.npy'),
                    allow_pickle=True)

                sensitivities = []
                specificities = []
                f1_scores = []
                aucs = []
                mccs = []

                with open(os.path.join(analyse_path, name + f'cp-{cp:04d}' + '.csv'), 'w',
                          newline='') as csvfile_cp:
                    fieldnames_cp = ['time',
                                     'accuracy',
                                     'sensitivity_cancer',
                                     'sensitivity_eso',
                                     'sensitivity_stomach',
                                     'sensitivity_background',
                                     'specificity_cancer',
                                     'specificity_eso',
                                     'specificity_stomach',
                                     'specificity_background',
                                     'f1_cancer',
                                     'f1_eso',
                                     'f1_stomach',
                                     'f1_background',
                                     'auc_cancer',
                                     'auc_eso',
                                     'auc_stomach',
                                     'auc_background',
                                     'mcc_cancer',
                                     'mcc_eso',
                                     'mcc_stomach',
                                     'mcc_background'
                                     ]
                    writer_cp = csv.DictWriter(csvfile_cp, fieldnames=fieldnames_cp)

                    writer_cp.writerow({'time': 'time',
                                        'accuracy': 'accuracy',
                                        'sensitivity_cancer': 'sensitivity_cancer',
                                        'sensitivity_eso': 'sensitivity_eso',
                                        'sensitivity_stomach': 'sensitivity_stomach',
                                        'sensitivity_background':'sensitivity_background',
                                        'specificity_cancer': 'specificity_cancer',
                                        'specificity_eso': 'specificity_eso',
                                        'specificity_stomach': 'specificity_stomach',
                                        'specificity_background':'specificity_background',
                                        'f1_cancer': 'f1_cancer',
                                        'f1_eso': 'f1_eso',
                                        'f1_stomach': 'f1_stomach',
                                        'f1_background':'f1_background',
                                        'auc_cancer': 'auc_cancer',
                                        'auc_eso': 'auc_eso',
                                        'auc_stomach': 'auc_stomach',
                                        'auc_background':'auc_background',
                                        'mcc_cancer': 'mcc_cancer',
                                        'mcc_eso': 'mcc_eso',
                                        'mcc_stomach': 'mcc_stomach',
                                        'mcc_background':'mcc_background'
                                        })

                    for i in range(gt_s.shape[0]):
                        print('Number', i + 1)
                        gt = np.array(gt_s[i])
                        print('gt', gt.shape, np.unique(gt), gt.dtype)
                        pred = np.array(pred_s[i])
                        pred = np.argmax(pred, axis=1)
                        # print(pred.shape, np.unique(pred), pred.dtype)

                        acc = accuracy_score(gt, pred)
                        spec = specificity(gt, pred)
                        sens = sensitivity(gt, pred)
                        f1_score_ = f1_score(gt, pred, average=None)
                        print(np.unique(gt))
                        if np.unique(gt).shape[0] > 2:
                            auc_ = roc_auc_score(np.eye(4)[gt], np.array(pred_s[i]), average=None)
                        else:
                            auc_ = [None, None, None]
                        # mcc_ = matthews_corrcoef(gt, pred)
                        mcc_ = mcc(gt, pred)

                        sensitivities.append(sens)
                        specificities.append(spec)
                        f1_scores.append(f1_score_)
                        aucs.append(auc_)
                        mccs.append(mcc_)

                        writer_cp.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                            'accuracy': str(acc).replace('.', ','),
                                            'sensitivity_cancer': str(sens[1]).replace('.', ','),
                                            'sensitivity_eso': str(sens[0]).replace('.', ','),
                                            'sensitivity_stomach': str(sens[2]).replace('.', ','),
                                            'sensitivity_background': str(sens[3]).replace('.', ','),
                                            'specificity_cancer': str(spec[1]).replace('.', ','),
                                            'specificity_eso': str(spec[0]).replace('.', ','),
                                            'specificity_stomach': str(spec[2]).replace('.', ','),
                                            'specificity_background': str(spec[3]).replace('.', ','),
                                            'f1_cancer': str(f1_score_[1]).replace('.', ','),
                                            'f1_eso': str(f1_score_[0]).replace('.', ','),
                                            'f1_stomach': str(f1_score_[2]).replace('.', ','),
                                            'f1_background': str(f1_score_[3]).replace('.', ','),
                                            'auc_cancer': str(auc_[1]).replace('.', ','),
                                            'auc_eso': str(auc_[0]).replace('.', ','),
                                            'auc_stomach': str(auc_[2]).replace('.', ','),
                                            'auc_background': str(auc_[3]).replace('.', ','),
                                            'mcc_cancer': str(mcc_[1]).replace('.', ','),
                                            'mcc_eso': str(mcc_[0]).replace('.', ','),
                                            'mcc_stomach': str(mcc_[2]).replace('.', ','),
                                            'mcc_background': str(mcc_[3]).replace('.', ',')
                                            })

                        print('Specificity', spec)
                        print('Sensitivity', sens)

                        print('------')

                    '''writer_cp.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), 
                                          'accuracy':acc, 
                                          'sensitivity_eso': sens[0],
                                          'sensitivity_cancer': sens[1],
                                          'sensitivity_stomach': sens[2],
                                          'specificity_eso': spec[0], 
                                          'specificity_cancer': spec[1], 
                                          'specificity_stomach': spec[2], 
                                          'f1_eso': f1_score_[0], 
                                          'f1_cancer':f1_score_[1], 
                                          'f1_stomach':f1_score_[2], 
                                          'auc_eso':auc_[0], 
                                          'auc_cancer': auc_[1], 
                                          'auc_stomach': auc_[2], 
                                          'mcc_eso': mcc_[0], 
                                          'mcc_specificity_cancer': mcc_[1], 
                                          'mcc_specificity_stomach': mcc_[2]})'''

                sensitivity_median = np.nanmedian(sensitivities, axis=0)
                specificity_median = np.nanmedian(specificities, axis=0)
                sensitivity_mean = np.nanmean(sensitivities, axis=0)
                specificity_mean = np.nanmean(specificities, axis=0)

                writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'checkpoint': str(cp),
                                 'sensitivity_mean': str(sensitivity_mean),
                                 'specificity_mean': str(specificity_mean)})
        
    def validate_checkpoints(self, name, 
                             cp_rng, 
                             mode='w'):
        
        analyse_path = os.path.join(self.test_preffix, name, 'analyse')
        if not os.path.exists(analyse_path):
            os.mkdir(analyse_path)

        with open(os.path.join(analyse_path, name + '.csv'), mode, newline='') as csvfile:
            fieldnames = ['time', 'checkpoint', 'sensitivity_mean', 'specificity_mean']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'time':'Time',
                             'checkpoint':'Checkpoint',
                             'sensitivity_mean':'sensitivity_mean',
                             'specificity_mean':'specificity_mean'})

            #for cp in tqdm(np.linspace(cp_rng[0], cp_rng[1], cp_rng[2])):
            for cp in tqdm(cp_rng):
                print('-----------------Checkpoint: '+str(cp)+'------------------- ')
                cp = int(cp)
                
                gt_s = np.load(os.path.join(self.test_preffix, name, f'cp-{cp:04d}', 'gt_by_patient.npy'), allow_pickle=True)
                pred_s = np.load(os.path.join(self.test_preffix, name, f'cp-{cp:04d}', 'predictions_by_patient.npy'), allow_pickle=True)

                sensitivities = []
                specificities = []
                f1_scores = []
                aucs = []
                mccs = []

                
                with open(os.path.join(analyse_path, name + f'cp-{cp:04d}' + '.csv'), 'w', newline='') as csvfile_cp:
                    fieldnames_cp = ['time', 
                                  'accuracy', 
                                  'sensitivity_cancer',
                                  'sensitivity_eso',
                                  'sensitivity_stomach',
                                  'specificity_cancer', 
                                  'specificity_eso', 
                                  'specificity_stomach', 
                                  'f1_cancer',
                                  'f1_eso',  
                                  'f1_stomach', 
                                  'auc_cancer', 
                                  'auc_eso', 
                                  'auc_stomach', 
                                  'mcc_cancer', 
                                  'mcc_eso', 
                                  'mcc_stomach' 
                                    ]
                    writer_cp = csv.DictWriter(csvfile_cp, fieldnames=fieldnames_cp)

                    writer_cp.writerow({'time':'time', 
                                          'accuracy':'accuracy', 
                                          'sensitivity_cancer':'sensitivity_cancer',
                                          'sensitivity_eso':'sensitivity_eso',
                                          'sensitivity_stomach':'sensitivity_stomach',
                                          'specificity_cancer':'specificity_cancer', 
                                          'specificity_eso':'specificity_eso', 
                                          'specificity_stomach':'specificity_stomach',                                
                                          'f1_cancer':'f1_cancer', 
                                          'f1_eso':'f1_eso', 
                                          'f1_stomach':'f1_stomach', 
                                          'auc_cancer': 'auc_cancer', 
                                          'auc_eso':'auc_eso', 
                                          'auc_stomach':'auc_stomach',
                                          'mcc_cancer':'mcc_cancer', 
                                          'mcc_eso':'mcc_eso', 
                                          'mcc_stomach':'mcc_stomach'
                                       })

                    for i in range(gt_s.shape[0]):
                        print('Number', i+1)
                        gt = np.array(gt_s[i])
                        print('gt', gt.shape, np.unique(gt), gt.dtype)
                        pred = np.array(pred_s[i])
                        pred = np.argmax(pred, axis=1)
                        #print(pred.shape, np.unique(pred), pred.dtype)

                        acc = accuracy_score(gt, pred)
                        spec = specificity(gt, pred)
                        sens = sensitivity(gt, pred)
                        f1_score_ = f1_score(gt, pred, average=None)
                        print( np.unique(gt))
                        if np.unique(gt).shape[0] > 2:
                            auc_ = roc_auc_score(np.eye(3)[gt], np.array(pred_s[i]), average=None)
                        else:
                            auc_ = [None, None, None]
                        #mcc_ = matthews_corrcoef(gt, pred)
                        mcc_ = mcc(gt, pred)

                        sensitivities.append(sens)
                        specificities.append(spec)
                        f1_scores.append(f1_score_)
                        aucs.append(auc_)
                        mccs.append(mcc_)


                        writer_cp.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), 
                                      'accuracy':str(acc).replace('.', ','), 
                                      'sensitivity_cancer': str(sens[1]).replace('.', ','),
                                      'sensitivity_eso':  str(sens[0]).replace('.', ','),
                                      'sensitivity_stomach':  str(sens[2]).replace('.', ','),
                                      'specificity_cancer':  str(spec[1]).replace('.', ','), 
                                      'specificity_eso':  str(spec[0]).replace('.', ','), 
                                      'specificity_stomach':  str(spec[2]).replace('.', ','), 
                                      'f1_cancer':  str(f1_score_[1]).replace('.', ','), 
                                      'f1_eso': str(f1_score_[0]).replace('.', ','), 
                                      'f1_stomach': str(f1_score_[2]).replace('.', ','), 
                                      'auc_cancer': str(auc_[1]).replace('.', ','), 
                                      'auc_eso':str(auc_[0]).replace('.', ','), 
                                      'auc_stomach': str(auc_[2]).replace('.', ','), 
                                      'mcc_cancer': str(mcc_[1]).replace('.', ','), 
                                      'mcc_eso': str(mcc_[0]).replace('.', ','), 
                                      'mcc_stomach': str(mcc_[2]).replace('.', ',')
                                           })

                        print('Specificity', spec)
                        print('Sensitivity', sens)

                        print('------')



                    '''writer_cp.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), 
                                          'accuracy':acc, 
                                          'sensitivity_eso': sens[0],
                                          'sensitivity_cancer': sens[1],
                                          'sensitivity_stomach': sens[2],
                                          'specificity_eso': spec[0], 
                                          'specificity_cancer': spec[1], 
                                          'specificity_stomach': spec[2], 
                                          'f1_eso': f1_score_[0], 
                                          'f1_cancer':f1_score_[1], 
                                          'f1_stomach':f1_score_[2], 
                                          'auc_eso':auc_[0], 
                                          'auc_cancer': auc_[1], 
                                          'auc_stomach': auc_[2], 
                                          'mcc_eso': mcc_[0], 
                                          'mcc_specificity_cancer': mcc_[1], 
                                          'mcc_specificity_stomach': mcc_[2]})'''

                sensitivity_median = np.nanmedian(sensitivities, axis=0)
                specificity_median = np.nanmedian(specificities, axis=0)
                sensitivity_mean = np.nanmean(sensitivities, axis=0)
                specificity_mean = np.nanmean(specificities, axis=0)

                writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'checkpoint':str(cp),
                                 'sensitivity_mean':str(sensitivity_mean),
                                 'specificity_mean':str(specificity_mean)})
            
if __name__ == '__main__':
    mat_validator = MatValidator()
    mat_validator.validate_checkpoints_brain('DatabaseBrainFMed', [40])
    mat_validator.validate_checkpoints_brain('DatabaseBrainMM', [36])
    mat_validator.validate_checkpoints_brain('DatabaseBrainSNV', [38])
    #mat_validator.validate_checkpoints('Esophagus_MedFilter', [38])
    #mat_validator.validate_checkpoints('Esophagus_SNV', [36])
    #mat_validator.validate_checkpoints('EsophagusDatabase', [38])
    
    