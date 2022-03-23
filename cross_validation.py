import config
from provider import get_trainer, get_data_loader
import numpy as np
from tqdm import tqdm
import glob
import test
import os
import csv
import datetime
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

import utils
from validator import Validator
from data_utils.data_loaders.data_loader_base import DataLoader


class CrossValidator:
    def __init__(self):
        self.test_path = config.TEST_NPZ_PATH

    def save_metrics_for_threshold(self, npy_folder, threshold, senss, specs, dices, aucs, thress, mcc_s, acc_s):
        predictions_by_patient = np.load(os.path.join(npy_folder, 'predictions_by_patient.npy'), allow_pickle=True)
        gt_by_patient = np.load(os.path.join(npy_folder, 'gt_by_patient.npy'), allow_pickle=True)

        with open(os.path.join(npy_folder, 'metrics_by_threshold_'+str(threshold)+'.csv'), 'w', newline='') as csvfile:
            fieldnames = ['time', 
                          'threshold', 
                          'name',
                          'acc',
                          'sensitivity', 
                          'specificity', 
                          'dice_0',
                          'dice_1',
                          'auc', 
                          'best_threshold',
                          'mcc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({'time': "Time",
                             'name': "Name",
                             'threshold': 'Threshold',
                             'acc': 'accuracy',
                             'sensitivity': 'Sensitivity',
                             'specificity': 'Specificity',
                             'dice_0': 'Dice 0',
                             'dice_1': 'Dice 1',
                             'auc': 'AUC',
                             'best_threshold': 'Best threshold',
                             'mcc': 'MCC'})

            # loading of names
            data = []
            with open(self.results_file, newline='') as csv_file:
                report_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
                for row in tqdm(report_reader):
                    data.append(row)
            names = np.array(data)[:, 4]

            for name, sn, sp, d, a, t, m, ac in zip(names, senss, specs, dices, aucs, thress, mcc_s, acc_s):
                writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold': str(threshold),
                                 'name': name,
                                 'acc': str(ac).replace('.', ','),
                                 'sensitivity': str(sn).replace('.', ','),
                                 'specificity': str(sp).replace('.', ','),
                                 'dice_0': str(d[0]).replace('.', ','),
                                 'dice_1': str(d[1]).replace('.', ','),
                                 'auc': str(a).replace('.', ','),
                                 'best_threshold': str(t).replace('.', ','),
                                 'mcc': str(m).replace('.', ',')})

            '''for patient in range(predictions_by_patient.shape[0]):
                predictions = np.where(np.array(predictions_by_patient[patient]) > threshold, 1, 0)

                sensitivity_p, specificity_p, dice_p = test.Tester.count_metrics(np.rint(gt_by_patient[patient]), 
                                                                                 predictions, 
                                                                                 str(threshold), 
                                                                                 save_stats=False, 
                                                                                 return_dice=True)
                sensitivities.append(sensitivity_p)
                specificities.append(specificity_p)
                dices.append(dice_p)

                writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold':str(threshold),
                                 'sensitivity':str(sensitivity_p),
                                 'specificity':str(specificity_p),
                                'dice':str(dice_p)})'''

            mcc_s, dices = np.array(mcc_s), np.array(dices)
            mcc_s[mcc_s == 0] = float('NaN')
            dices[dices == 0] = float('NaN')
            sensitivity_median = np.nanmedian(senss)
            specificity_median = np.nanmedian(specs)
            dice_median = np.nanmedian(dices, axis=0)
            auc_median = np.nanmedian(aucs)
            thr_median = np.nanmedian(thress)
            mcc_median = np.nanmedian(mcc_s)
            acc_median = np.nanmedian(acc_s)

            sensitivity_mean = np.nanmean(senss)
            specificity_mean = np.nanmean(specs)
            dice_mean = np.nanmean(dices, axis=0)
            auc_mean = np.nanmean(aucs)
            thr_mean = np.nanmean(thress)
            mcc_mean = np.nanmean(mcc_s)
            acc_mean = np.nanmean(acc_s)

            writer.writerow({'time': "TOTAL MEAN",
                             'threshold': str(threshold),
                             'name': '-',
                             'acc': str(acc_mean).replace('.', ','),
                             'sensitivity': str(sensitivity_mean).replace('.', ','),
                             'specificity': str(specificity_mean).replace('.', ','),
                             'dice_0': str(dice_mean[0]).replace('.', ','),
                             'dice_1': str(dice_mean[1]).replace('.', ','),
                             'auc': str(auc_mean).replace('.', ','),
                             'best_threshold': str(thr_mean).replace('.', ','),
                             'mcc': str(mcc_mean).replace('.', ',')})

            writer.writerow({'time': "TOTAL MEDIAN",
                             'threshold': str(threshold),
                             'name': '-',
                             'acc': str(acc_median),
                             'sensitivity': str(sensitivity_median),
                             'specificity': str(specificity_median),
                             'dice_0': str(dice_median[0]).replace('.', ','),
                             'dice_1': str(dice_median[1]).replace('.', ','),
                             #'dice': str(dice_median),
                             'auc': str(auc_median),
                             'best_threshold': str(thr_median),
                             'mcc': str(mcc_median)})

            writer.writerow({'time': "STD",
                             'threshold': str(threshold),
                             'name': '-',
                             'acc': str(np.nanstd(acc_s)),
                             'sensitivity': str(np.nanstd(senss)),
                             'specificity': str(np.nanstd(specs)),
                             #'dice': str(np.nanstd(dices)),
                             'dice_0': str(np.nanstd(dices, axis=0)[0]).replace('.', ','),
                             'dice_1': str(np.nanstd(dices, axis=0)[1]).replace('.', ','),
                             'auc': np.nanstd(aucs),
                             'best_threshold': str(np.nanstd(thress)),
                             'mcc': str(np.nanstd(mcc_s))})

    def count_metrics_on_diff_thresholds(self, npy_folder,
                                         threshold_range_params=[0, 1, 21],
                                         threshold_range_plain=None):
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
            mcc_s = []
            acc_s = []
            for patient in range(predictions_by_patient.shape[0]):
                fpr, tpr, thresholds_p = metrics.roc_curve(np.rint(gt_by_patient[patient]),
                                                           np.array(predictions_by_patient[patient]))
                roc_auc = metrics.auc(fpr, tpr)
                predictions = np.array(np.array(predictions_by_patient[patient]) > threshold).astype(np.uint8)

                threshold_p = thresholds_p[np.argmax(tpr - fpr)]
                gt = np.rint(gt_by_patient[patient])

                sensitivity_p, specificity_p, dice_p = test.Tester.count_metrics(gt,
                                                                                 predictions,
                                                                                 str(threshold),
                                                                                 save_stats=False,
                                                                                 return_dice=True)
                mcc_p = metrics.matthews_corrcoef(gt, predictions)
                acc_p = metrics.accuracy_score(gt, predictions)

                sens.append(sensitivity_p)
                spec.append(specificity_p)
                dices.append(dice_p)
                thresholds.append(threshold_p)
                aucs.append(roc_auc)
                mcc_s.append(mcc_p)
                acc_s.append(acc_p)

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
            self.save_metrics_for_threshold(npy_folder, threshold, sens, spec, dices, aucs, thresholds, mcc_s, acc_s)

            path_metrics = os.path.join(npy_folder, 'metrics_threshold_relation_by_patient.csv')
            with open(path_metrics, 'a', newline='') as csvfile:
                fieldnames = ['time', 
                              'threshold', 
                              'sensitivity_median', 
                              'specificity_median',
                              'sensitivity_mean', 
                              'specificity_mean', 
                              'dice_median', 
                              'dice_mean']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold': str(threshold),
                                 'sensitivity_median': str(sensitivity_median),
                                 'specificity_median': str(specificity_median),
                                 'sensitivity_mean': str(sensitivity_mean),
                                 'specificity_mean': str(specificity_mean),
                                 'dice_median': str(np.nanmedian(dices)),
                                'dice_mean': str(np.nanmean(dices))})

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
        """

        :param csv_path:
        0 - date
        1 - index
        2 - sensitivity
        3 - specificity
        4 - .dat path
        5 - model path
        """

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

                model_path = row[5]
                if 'LOCAL' in config.MODE:
                    model_path = row[5].split('hsi-experiments')[-1][1:]
                
                tester = test.Tester(checkpoint, ['data'], '', MODEL_FOLDER=model_path)
                
                name = DataLoader.get_name_easy(row[4], delimiter='/')

                #sensitivity, specificity = tester.test_one_image(row[4],
                sensitivity, specificity = tester.test_one_image(os.path.join(self.test_path, name + ".npz"),
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
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
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
        """

        :param csv_path:
        0 - date
        1 - index
        2 - sensitivity
        3 - specificity
        4 - .dat path
        5 - model path
        """
        with open(csv_path, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                print(', '.join(row))

                #tester = test.Tester('cp-00'+str(config.EPOCHS), ['data'], save_path, MODEL_FOLDER=row[5])
                if int(row[1]) in [0, 1, 2, 3]:
                    tester = test.Tester('cp-0200', ['data'], save_path, MODEL_FOLDER=row[5])
                else:
                    tester = test.Tester( f'cp-{config.EPOCHS:04d}', ['data'], save_path, MODEL_FOLDER=row[5])

                name = DataLoader.get_name_easy(row[4])
                #sensitivity, specificity = tester.test_one_image(row[4],
                sensitivity, specificity = tester.test_one_image(os.path.join(config.RAW_NPZ_PATH, name + ".npz"),
                                                                 path_image=row[4] + '_Mask JW Kolo.png', save=True,
                                                                 show=False, test_all_spectra=test_all_spectra,
                                                                 saving_path='', spectrum_shift=0)

    def cross_validation(self, root_folder_name, csv_filename=None):
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
        csv_filename = os.path.join(root_folder, csv_filename)

        for ind, indexes in enumerate(splits):                
            old_model_name = config.MODEL_NAME
            indexes = np.array(indexes[0])
            
            if len(indexes) > 1:
                for i in indexes:
                    config.MODEL_NAME += '_' + str(i)
            else:
                config.MODEL_NAME += '_' + str(ind) + '_' + DataLoader.get_name_easy(np.array(paths)[indexes][0])
                # скопировала на всякий случай с сервера, когда-то тут была ошибка
                #config.MODEL_NAME += '_' + str(indexes[0]) + '_' + np.array(paths)[indexes][0].split("/")[-1].split(".")[0].split('SpecCube')[0]

            trainer = get_trainer(except_indexes=[DataLoader.get_name_easy(p) for p in np.array(paths)[indexes]])
            model = trainer.train()
            for i in indexes:
                name = DataLoader.get_name_easy(paths[i])

                tester = test.Tester( f'cp-{config.EPOCHS:04d}',
                                      ['data'],
                                      '',
                                      LOGS_PATH=root_folder,
                                      MODEL_NAME=config.MODEL_NAME.split(config.SYSTEM_PATHS_DELIMITER)[-1])

                #sensitivity, specificity = tester.test_one_image(paths[i],
                sensitivity, specificity = 0, 0
                '''sensitivity, specificity = tester.test_one_image(os.path.join(config.RAW_NPY_PATH, name + ".npz"),
                                    path_image=paths[i] + '_Mask JW Kolo.png',
                                    save=False,
                                    show=False,
                                    test_all_spectra=False,
                                    save_stats=False,
                                    folder_name=config.MODEL_NAME,
                                    test_batch=True,  #be careful!
                                    spectrum_shift=0)   #be careful!'''

                print('For path=', paths[i], ', (index: ', str(i), ') sensitivity= ', str(sensitivity), ' specificity= ', str(specificity), ' MODEL_NAME = ', config.MODEL_NAME)

                with open(csv_filename,'a', newline='') as csvfile: # for full cross_valid and for separate file
                    fieldnames = ['time', 'index', 'sensitivity', 'specificity', 'name', 'model_name']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                     'index':str(i),
                                     'sensitivity':str(sensitivity),
                                     'specificity':str(specificity),
                                     'name': paths[i],
                                     'model_name': config.MODEL_NAME})

            config.MODEL_NAME = old_model_name
        return csv_filename

    def get_history(self, model_path):
        history_paths = utils.glob_multiple_file_types(model_path, '.*.npy', '*.npy')
        if len(history_paths) == 0:
            raise ValueError('Error! No history files were found!')
        if len(history_paths) > 1:
            raise ValueError(f'Error! Too many .npy files were found in {model_path}!')

        history_path = history_paths[0]
        history = np.load(history_path, allow_pickle=True)
        if len(history.shape) == 0:
            history = history.item()
        return history

    def get_best_checkpoint_from_valid(self, results_file):
        model_paths = []
        best_checkpoints = []

        with open(results_file, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                model_path = row[5]
                if 'LOCAL' in config.MODE:
                    model_path = row[5].split('hsi-experiments')[-1][1:]

                model_paths.append(model_path)

                history = self.get_history(model_path)

                best_checkpoint = np.argmin(history[config.HISTORY_ARGMIN])
                best_checkpoints.append(best_checkpoint)
         
        best_checkpoint = utils.round_to_the_nearest_even_int(np.median(best_checkpoints))
            
        return best_checkpoint, best_checkpoints, model_paths

    def save_ROC_thresholds_for_checkpoint(self, checkpoint,
                                           save_path_,
                                           csv_file,
                                           thr_ranges=[],
                                           execution_flags=[True]):
        if type(checkpoint) == int:
            checkpoint = f'cp-{checkpoint:04d}'
        
        self.results_file = csv_file
        print('CHECKPOINT: ', checkpoint)

        if not os.path.exists(save_path_):
            os.mkdir(save_path_)

        save_path = os.path.join(save_path_, checkpoint)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if execution_flags[0]:
            self.count_ROC(csv_file, save_path, checkpoint=checkpoint)

        if len(thr_ranges) == 0:
            print('WARNING! No thresholds were given!')

        for rng in thr_ranges:
            self.count_metrics_on_diff_thresholds(save_path, threshold_range_params=rng)

        # self.count_metrics_on_diff_thresholds(save_path, threshold_range_params=[0.0001, 0.0009, 9])
        # self.count_metrics_on_diff_thresholds(save_path, threshold_range_params=[0.05, 0.25, 8])
        # self.count_metrics_on_diff_thresholds(save_path, threshold_range_params=[0.001, 0.04, 8])
        # self.count_metrics_on_diff_thresholds(save_path, threshold_range_params=[0.05, 0.5, 5])

    def compare_checkpoints(self, rng, save_path_, results_file):
        rg = np.linspace(rng[0], rng[1], rng[2]).astype(int)
        checkpoints = [f'cp-{i:04d}' for i in rg]
        print('Checkpoints: ', checkpoints)

        for checkpoint in tqdm(checkpoints):
            self.save_ROC_thresholds_for_checkpoint(checkpoint, save_path_, results_file)

    def cross_validation_spain(self):
        #prefix = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments'
        prefix = 'C:\\Users\\tkachenko\\Desktop\\HSI\\'

        name = config.bea_db

        #csv_path = cross_validator.cross_validation(name)

        if name == 'ColonDatabase':
            csv_path = 'C:\\Users\\tkachenko\\Desktop\\HSI\\hsi-experiments\\logs\\CV_3d_bea_colon_sample_weights_1output\\CV_3d_bea_colon_sample_weights_1output_stats_16.12.2021-13_09_52.csv'
        else:
            csv_path = glob.glob(os.path.join(prefix, 'hsi-experiments', 'logs', name + '*', '*.csv'))[0]
        print(csv_path)

        best_checkpoint, best_checkpoints, model_paths = cross_validator.get_best_checkpoint_from_valid(csv_path)
        print(best_checkpoint)
        print(best_checkpoints)
        print(model_paths)

        test_path = os.path.join('test', name)
        test_path_whole = os.path.join('test', name+'_whole_image')
        if config.MODE == 'CLUSTER':
            test_path = os.path.join(prefix, test_path)
            test_path_whole = os.path.join(prefix, test_path_whole)

        #save predictions for whole image
        self.test_path = config.TEST_NPZ_PATH
        cross_validator.save_ROC_thresholds_for_checkpoint(best_checkpoint,
                                                           test_path_whole,
                                                           csv_path,
                                                           thr_ranges=[],
                                                           execution_flags=[False])

        #save annotated predictions
        self.test_path = config.RAW_NPZ_PATH
        cross_validator.save_ROC_thresholds_for_checkpoint(best_checkpoint,
                                                           test_path,
                                                           csv_path,
                                                           #thr_ranges=[],
                                                           thr_ranges=[
                                                               #[0.001, 0.009, 10],
                                                                       [0.01, 0.09, 10],
                                                                       #[0.1, 0.6, 10],
                                                               #[0.15, 0.25, 10]
                                                           ],
                                                           execution_flags=[False])


if __name__ =='__main__':
    
    try:
        cross_validator = CrossValidator()
        cross_validator.cross_validation_spain()
        '''prefix = '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments'
        
        name = config.bea_db
        #name = 'CV_3d_sample_weights_every_third'
        
        #csv_path = cross_validator.cross_validation(name)
        csv_path = glob.glob(os.path.join('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/', name, '*.csv'))[0]
        
        best_checkpoint, best_checkpoints, model_paths = cross_validator.get_best_checkpoint_from_valid(csv_path)
        print(best_checkpoint)
        print(best_checkpoints)
        print(model_paths)

        test_path = os.path.join('test', name)
        if config.MODE == 'CLUSTER':
            test_path = os.path.join(prefix, test_path)
        #test_path = os.path.join(prefix, 'test', name)
        #cross_validator.compare_checkpoints([2, 20, 10], test_path, csv_path)
        
        cross_validator.save_ROC_thresholds_for_checkpoint(best_checkpoint,
                                                           test_path,
                                                           csv_path,
                                                           thr_ranges=[],
                                                           execution_flags=[True])'''
        
        #validator = Validator()
        #validator.find_best_checkpoint(test_path)
        
        #utils.send_tg_message('Mariia, operations in cross_validation.py are successfully completed!')
               
    except Exception as e:
        print(e)

        #utils.send_tg_message(f'Mariia, ERROR!!!, In CV error {e}')
        
        raise e
