import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(1, os.path.join(currentdir, 'utils')) 
sys.path.insert(2, os.path.join(currentdir, 'data_utils')) 
sys.path.insert(2, os.path.join(currentdir, 'models')) 

print('paths from cv', sys.path)

import config
from train import train
import numpy as np
from tqdm import tqdm
import glob
import test
import os
import csv
import datetime
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from utils import send_tg_message
from validator import Validator


class CrossValidator():

    def save_metrics_for_threshold(self, npy_folder, threshold, senss, specs, dices, aucs, thress):
        predictions_by_patient = np.load(os.path.join(npy_folder, 'predictions_by_patient.npy'), allow_pickle=True)
        gt_by_patient = np.load(os.path.join(npy_folder, 'gt_by_patient.npy'), allow_pickle=True)

        with open(os.path.join(npy_folder, 'metrics_by_threshold_'+str(threshold)+'.csv'), 'w', newline='') as csvfile:
            fieldnames = ['time', 
                          'threshold', 
                          'name', 
                          'sensitivity', 
                          'specificity', 
                          'dice', 
                          'auc', 
                          'best_threshold']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({'time':"Time",
                             'name': "Name",
                             'threshold':'Threshold',
                             'sensitivity':'Sensitivity',
                             'specificity':'Specificity',
                             'dice':'Dice',
                             'auc':'AUC', 
                             'best_threshold':'Best threshold'})

            #loading of names
            data = []
            with open(self.results_file, newline='') as csvfile:
                report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in tqdm(report_reader):
                    data.append(row)
            names = np.array(data)[:, 4]

            for name, sn, sp, d, a, t in zip(names, senss, specs, dices, aucs, thress):
                writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold':str(threshold),
                                 'name': name,
                                 'sensitivity':str(sn),
                                 'specificity':str(sp),
                                 'dice':str(d),
                                 'auc':str(a),
                                 'best_threshold':str(t)})


            '''for patient in range(predictions_by_patient.shape[0]):
                predictions = np.where(np.array(predictions_by_patient[patient]) > threshold, 1, 0)

                sensitivity_p, specificity_p, dice_p = test.Tester.count_metrics(np.rint(gt_by_patient[patient]), predictions, str(threshold), save_stats=False, return_dice=True)
                sensitivities.append(sensitivity_p)
                specificities.append(specificity_p)
                dices.append(dice_p)

                writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold':str(threshold),
                                 'sensitivity':str(sensitivity_p),
                                 'specificity':str(specificity_p),
                                'dice':str(dice_p)})'''

            sensitivity_median = np.nanmedian(senss)
            specificity_median = np.nanmedian(specs)
            dice_median = np.nanmedian(dices)
            auc_median = np.nanmedian(aucs)
            thr_median = np.nanmedian(thress)

            sensitivity_mean = np.nanmean(senss)
            specificity_mean = np.nanmean(specs)
            dice_mean = np.nanmean(dices)
            auc_mean = np.nanmean(aucs)
            thr_mean = np.nanmean(thress)

            writer.writerow({'time':"GESAMT MEAN",
                             'threshold':str(threshold),
                             'name': '-',
                             'sensitivity':str(sensitivity_mean),
                             'specificity':str(specificity_mean),
                             'dice':str(dice_mean),
                             'auc':str(auc_mean),
                             'best_threshold':str(thr_mean)})

            writer.writerow({'time':"GESAMT MEDIAN",
                             'threshold':str(threshold),
                             'name': '-',
                             'sensitivity':str(sensitivity_median),
                             'specificity':str(specificity_median),
                             'dice':str(dice_median),
                             'auc':str(auc_median),
                             'best_threshold':str(thr_median)})

            writer.writerow({'time':"STD",
                             'threshold':str(threshold),
                             'name': '-',
                             'sensitivity':str(np.nanstd(senss)),
                             'specificity':str(np.nanstd(specs)),
                             'dice':str(np.nanstd(dices)),
                             'auc':np.nanstd(aucs),
                             'best_threshold':str(np.nanstd(thress))})




    def count_metrics_on_diff_thresholds(self, npy_folder, threshold_range_params=[0, 1, 21], threshold_range_plain=None):
        predictions_by_patient = np.load(os.path.join(npy_folder, 'predictions_by_patient.npy'), allow_pickle=True)
        gt_by_patient = np.load(os.path.join(npy_folder, 'gt_by_patient.npy'), allow_pickle=True)

        sensitivities_median = []
        specificities_median = []
        sensitivities_mean = []
        specificities_mean = []

        rng = np.round(np.linspace(threshold_range_params[0], threshold_range_params[1], threshold_range_params[2]), 4)
        if threshold_range_plain is not None:
            rng = threshold_range_plain.copy()

        for threshold in tqdm(rng):
            sens = []
            spec = []
            dices = []
            thresholds = []
            aucs = []
            for patient in range(predictions_by_patient.shape[0]):
                fpr, tpr, thresholds_p = metrics.roc_curve(np.rint(gt_by_patient[patient]), np.array(predictions_by_patient[patient]))
                roc_auc = metrics.auc(fpr, tpr)
                predictions = np.array(np.array(predictions_by_patient[patient]) > threshold).astype(np.uint8)

                threshold_p = thresholds_p[np.argmax(tpr - fpr)]

                sensitivity_p, specificity_p, dice_p = test.Tester.count_metrics(np.rint(gt_by_patient[patient]), predictions, str(threshold), save_stats=False, return_dice=True)
                sens.append(sensitivity_p)
                spec.append(specificity_p)
                dices.append(dice_p)
                thresholds.append(threshold_p)
                aucs.append(roc_auc)

            sensitivity_median = np.nanmedian(sens)
            specificity_median = np.nanmedian(spec)
            sensitivity_mean = np.nanmean(sens)
            specificity_mean = np.nanmean(spec)

            sensitivities_median.append(sensitivity_median)
            specificities_median.append(specificity_median)
            sensitivities_mean.append(sensitivity_mean)
            specificities_mean.append(specificity_mean)

            thresholds = np.array(thresholds)
            thresholds[thresholds > 1.] = np.nan
            self.save_metrics_for_threshold(npy_folder, threshold, sens, spec, dices, aucs, thresholds)

            with open(os.path.join(npy_folder, 'metrics_threshold_relation_by_patient.csv'), 'a', newline='') as csvfile:
                fieldnames = ['time', 
                              'threshold', 
                              'sensitivity_median', 
                              'specificity_median',
                              'sensitivity_mean', 
                              'specificity_mean', 
                              'dice_median', 
                              'dice_mean']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold':str(threshold),
                                 'sensitivity_median':str(sensitivity_median),
                                 'specificity_median':str(specificity_median),
                                 'sensitivity_mean':str(sensitivity_mean),
                                 'specificity_mean':str(specificity_mean),
                                 'dice_median':str(np.nanmedian(dices)),
                                'dice_mean':str(np.nanmean(dices))})

        plt.plot(rng, sensitivities_median)
        plt.plot(rng, specificities_median)
        plt.savefig(os.path.join(npy_folder, 'thresholds_metrics_curves_median.png'))
        #plt.show()
        plt.clf()

        plt.plot(rng, sensitivities_mean)
        plt.plot(rng, specificities_mean)
        plt.savefig(os.path.join(npy_folder, 'thresholds_metrics_curves_mean.png'))
        #plt.show()
        plt.clf()




    def count_ROC(self, csv_path, save_path, checkpoint=None, save_roc_auc_curve=False):
        '''

        :param csv_path:
        0 - date
        1 - index
        2 - sensitivity
        3 - specificity
        4 - .dat path
        5 - model path
        '''

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

                sensitivity = specificity = 0
                tester = None
                #!!!!!!!!!!!!!!!!!!!!!!!!REMOVE
                checkpoint_number = int(checkpoint.split('-')[-1])
                print(checkpoint_number)
                
                tester = test.Tester(checkpoint, ['data'], '', MODEL_FOLDER=row[5])
                
                name = row[4].split("/")[-1].split(".")[0].split('SpecCube')[0]

                #sensitivity, specificity = tester.test_one_image(row[4],
                sensitivity, specificity = tester.test_one_image(os.path.join(config.RAW_NPY_PATH, name + ".npz"),
                                      path_image=row[4] + '_Mask JW Kolo.png',
                                      save=False,
                                      show=False,
                                      test_all_spectra=False,
                                      save_stats=False,
                                      folder_name='', 
                                      test_batch=True,
                                      spectrum_shift=0)
                
                predictions_by_patient.append(tester.all_predictions_raw)
                gt_by_patient.append(tester.all_gt)
                
                if save_roc_auc_curve:
                    all_predictions_raw += tester.all_predictions_raw
                    all_gt += tester.all_gt

        #saving of predictions
        np.save(os.path.join(save_path, 'predictions_by_patient.npy'),  predictions_by_patient)
        np.save(os.path.join(save_path, 'gt_by_patient.npy'), gt_by_patient)

        #roc auc part
        if save_roc_auc_curve:
            fpr, tpr, threshold = metrics.roc_curve(all_gt, all_predictions_raw)
            print('threshold', threshold, threshold[np.argmax(tpr - fpr)])
            roc_auc = metrics.auc(fpr, tpr)

            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig(os.path.join(save_path, 'roc.png'))
            #plt.show()
            plt.clf()


    #old one, needs actualization
    def run_csv_and_save_images(self, csv_path, save_path, test_all_spectra=True):
        '''

        :param csv_path:
        0 - date
        1 - index
        2 - sensitivity
        3 - specificity
        4 - .dat path
        5 - model path
        '''
        with open(csv_path, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                print(', '.join(row))

                #tester = test.Tester('cp-00'+str(config.EPOCHS), ['data'], save_path, MODEL_FOLDER=row[5])
                if int(row[1]) in [0, 1, 2, 3]:
                    tester = test.Tester('cp-0200', ['data'], save_path, MODEL_FOLDER=row[5])
                else:
                    tester = test.Tester( f'cp-{config.EPOCHS:04d}', ['data'], save_path, MODEL_FOLDER=row[5])

                name = row[4].split("/")[-1].split(".")[0].split('SpecCube')[0]

                #sensitivity, specificity = tester.test_one_image(row[4],
                sensitivity, specificity = tester.test_one_image(os.path.join(config.RAW_NPY_PATH, name + ".npz"),
                                      path_image=row[4] + '_Mask JW Kolo.png',
                                      save=True,
                                      show=False,
                                      test_all_spectra=test_all_spectra,
                                      save_stats=False,
                                      spectrum_shift=0,
                                      test_batch=True,
                                      folder_name='')


    def cross_validation(self, root_folder_name):
        config.MODEL_NAME_PATHS.append(root_folder_name)

        root_folder = os.path.join(*config.MODEL_NAME_PATHS)
        config.MODEL_NAME = config.get_model_name(config.MODEL_NAME_PATHS)
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        paths = []
        for data_path in config.DATA_PATHS:
            paths += glob.glob(os.path.join(data_path, '*.dat'))

        csv_filename = os.path.join(root_folder, root_folder_name + '_stats'+ datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S") +'.csv')

        splits = np.array_split(range(len(paths)), config.CROSS_VALIDATION_SPLIT)

        for ind, indexes in enumerate(splits[-1:]):
            old_model_name = config.MODEL_NAME
            if len(indexes) > 1:
                for i in indexes:
                    config.MODEL_NAME += '_' + str(i)
            else:
                config.MODEL_NAME += '_' + str(ind) + '_' + np.array(paths)[indexes][0].split("/")[-1].split(".")[0].split('SpecCube')[0]

            model = train(paths=paths, except_indexes=[p.split("/")[-1].split(".")[0].split('SpecCube')[0] for p in np.array(paths)[indexes]])
            for i in indexes:
                name = paths[i].split("/")[-1].split(".")[0].split('SpecCube')[0]

                tester = test.Tester( f'cp-{config.EPOCHS:04d}', ['data'], '', LOGS_PATH=root_folder, MODEL_NAME=config.MODEL_NAME.split('\\')[-1])

                #sensitivity, specificity = tester.test_one_image(paths[i],
                sensitivity, specificity = 0, 0
                '''sensitivity, specificity = tester.test_one_image(os.path.join(config.RAW_NPY_PATH, name + ".npz"),
                                    path_image=paths[i] + '_Mask JW Kolo.png',
                                    save=False,
                                    show=False,
                                    test_all_spectra=False,
                                    save_stats=False,
                                    folder_name=config.MODEL_NAME,
                                    test_batch=True,  #be carefull!
                                    spectrum_shift=0)   #be carefull!'''

                print('For path=', paths[i], ', (index: ', str(i), ') sensitivity= ', str(sensitivity), ' specificity= ', str(specificity), ' MODEL_NAME = ', config.MODEL_NAME)

                with open(csv_filename,'a', newline='') as csvfile: # for full cross_valid and for separate file
                    fieldnames = ['time', 'index', 'sensitivity', 'specificity', 'name', 'model_name']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                     'index':str(i),
                                     'sensitivity':str(sensitivity),
                                     'specificity':str(specificity),
                                     'name':paths[i],
                                     'model_name': config.MODEL_NAME})

            config.MODEL_NAME = old_model_name
        return csv_filename

    def compare_checkpoints(self, rng, save_path_, results_file):
        rg = np.linspace(rng[0], rng[1],rng[2]).astype(int)
        checkpoints = [f'cp-{i:04d}' for i in rg]
        print('Checkpoints', checkpoints)
        self.results_file = results_file
        for checkpoint in tqdm(checkpoints):
            print(checkpoint)

            if not os.path.exists(save_path_):
                os.mkdir(save_path_)

            save_path = os.path.join(save_path_, checkpoint)

            if not os.path.exists(save_path):
                os.mkdir(save_path)

            #self.count_ROC(results_file, save_path, checkpoint=checkpoint)
             
            self.count_metrics_on_diff_thresholds(save_path, all=True, threshold_range_params=[0.0001, 0.0009, 9])
            #self.count_metrics_on_diff_thresholds(save_path, all=True, threshold_range_params=[0.05, 0.5, 10])
            #self.count_metrics_on_diff_thresholds(save_path, all=True, threshold_range_params=[0.001, 0.04, 10])
            #self.count_metrics_on_diff_thresholds(save_path, all=True, threshold_range_params=[0.15, 0.7, 12])
            #self.count_metrics_on_diff_thresholds(save_path, all=True, threshold_range_params=[0.05, 0.5, 5])

if __name__ =='__main__':
    
    try:
        cross_validator = CrossValidator()
        preffix = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments'
        
        name = 'CV_3d_inception'
        #name = 'CV_3d_sample_weights_every_third'
        
        csv_path = cross_validator.cross_validation(name)
        #csv_path = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d_bg_every_third/CV_3d_bg_every_third_stats_25.11.2021-00_10_04.csv'
        #csv_path = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d_svn_every_third/CV_3d_svn_every_third_stats_24.11.2021-15_58_11.csv'
        #csv_path = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d_every_third/CV_3d_every_third_stats_24.11.2021-15_25_23.csv'
        #csv_path = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d/CV_3d_stats_17.11.2021-21_16_05.csv'
        #csv_path = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d_sample_weights_every_third/CV_3d_sample_weights_every_third_stats_02.12.2021-22_32_48.csv'
        
        test_path = os.path.join(preffix, 'test', name)
        cross_validator.compare_checkpoints([2, 40, 20], test_path, csv_path)
        
        validator = Validator()
        validator.find_best_checkpoint(test_path)
        
        if config.TELEGRAM_SENDING:
            send_tg_message('Mariia, operations in cross_validation.py are successfully completed!')
               
    except Exception as e:
        print(e)

        if config.TELEGRAM_SENDING:
            send_tg_message(f'Mariia, ERROR!!!, In CV error {e}')
        
        raise e
