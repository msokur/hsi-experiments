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

def save_metrics_for_threshold(npy_folder, threshold):
    predictions_by_patient = np.load(os.path.join(npy_folder, 'predictions_by_patient.npy'), allow_pickle=True)
    gt_by_patient = np.load(os.path.join(npy_folder, 'gt_by_patient.npy'), allow_pickle=True)

    sensitivities = []
    specificities = []
    dices = []

    with open(os.path.join(npy_folder, 'metrics_by_threshold_'+str(threshold)+'.csv'), 'a', newline='') as csvfile:
        fieldnames = ['time', 'threshold', 'sensitivity', 'specificity', 'dice']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for patient in range(predictions_by_patient.shape[0]):
            predictions = np.where(np.array(predictions_by_patient[patient]) > threshold, 1, 0)

            sensitivity_p, specificity_p, dice_p = test.Tester.count_metrics(np.rint(gt_by_patient[patient]), predictions, str(threshold), save_stats=False, return_dice=True)
            sensitivities.append(sensitivity_p)
            specificities.append(specificity_p)
            dices.append(dice_p)

            writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                             'threshold':str(threshold),
                             'sensitivity':str(sensitivity_p),
                             'specificity':str(specificity_p),
                            'dice':str(dice_p)})

        sensitivity_median = np.nanmedian(sensitivities)
        specificity_median = np.nanmedian(specificities)
        dice_median = np.nanmedian(dices)
        
        sensitivity_mean = np.nanmean(sensitivities)
        specificity_mean = np.nanmean(specificities)
        dice_mean = np.nanmean(dices)

        writer.writerow({'time':"GESAMT MEDIAN",
                         'threshold':str(threshold),
                         'sensitivity':str(sensitivity_median),
                         'specificity':str(specificity_median),
                        'dice':str(dice_median)})
         
        writer.writerow({'time':"GESAMT MEAN",
                         'threshold':str(threshold),
                         'sensitivity':str(sensitivity_mean),
                         'specificity':str(specificity_mean),
                        'dice':str(dice_mean)})
        
        writer.writerow({'time':"STD",
                         'threshold':str(threshold),
                         'sensitivity':str(np.nanstd(sensitivities)),
                         'specificity':str(np.nanstd(specificities)),
                        'dice':str(np.nanstd(dices))})




def count_metrics_on_diff_thresholds(npy_folder, all=False, threshold_range_params=[0, 1, 21], threshold_range_plain=None):
    all_gt = np.load(os.path.join(npy_folder, 'all_gt.npy'))
    predictions_by_patient = np.load(os.path.join(npy_folder, 'predictions_by_patient.npy'), allow_pickle=True)
    all_predictions_raw = np.load(os.path.join(npy_folder, 'all_predictions_raw.npy'))
    gt_by_patient = np.load(os.path.join(npy_folder, 'gt_by_patient.npy'), allow_pickle=True)

    sensitivities_median = []
    specificities_median = []
    sensitivities_mean = []
    specificities_mean = []

    rng = np.round(np.linspace(threshold_range_params[0], threshold_range_params[1], threshold_range_params[2]), 2)
    if threshold_range_plain is not None:
        rng = threshold_range_plain.copy()

    for threshold in tqdm(rng):
        if all:
            sens = []
            spec = []
            dices = []
            thresholds = []
            aucs = []
            for patient in range(predictions_by_patient.shape[0]):
                fpr, tpr, thresholds_p = metrics.roc_curve(np.rint(gt_by_patient[patient]), np.array(predictions_by_patient[patient]))
                roc_auc = metrics.auc(fpr, tpr)
                predictions = np.where(np.array(predictions_by_patient[patient]) > threshold, 1, 0)

                threshold_p = thresholds_p[np.argmax(tpr - fpr)]

                sensitivity_p, specificity_p, dice_p = test.Tester.count_metrics(np.rint(gt_by_patient[patient]), predictions, str(threshold), save_stats=False, return_dice=True)
                sens.append(sensitivity_p)
                spec.append(specificity_p)
                dices.append(dice_p)
                thresholds.append(threshold_p)
                aucs.append(roc_auc)

            '''print('SENS')
            for i in sens:
                print(i)
            print(sens)
            print('sens mean, median, std', np.nanmean(sens), np.nanmedian(sens), np.nanstd(sens))
            print('SPEC')
            for i in spec:
                print(i)
            print('spec mean, median, std', np.nanmean(spec), np.nanmedian(spec), np.nanstd(spec))
            print('DICEs')

            for i in dices:
                print(i)
            dices[dices == 0.] = np.nan
            print('dices mean, median, std', np.nanmean(dices), np.nanmedian(dices), np.nanstd(dices))
            print('thresholds')
            thresholds = np.array(thresholds)
            for i in thresholds:
                print(i)
            thresholds[thresholds > 1.] = np.nan
            print('thresholds mean, median, std', np.nanmean(thresholds), np.nanmedian(thresholds), np.nanstd(thresholds))
            print('aucs')
            for i in aucs:
                print(i)
            print('aucs mean, median, std', np.nanmean(aucs), np.nanmedian(aucs), np.nanstd(aucs))'''


            sensitivity_median = np.nanmedian(sens)
            specificity_median = np.nanmedian(spec)
            sensitivity_mean = np.nanmean(sens)
            specificity_mean = np.nanmean(spec)

            sensitivities_median.append(sensitivity_median)
            specificities_median.append(specificity_median)
            sensitivities_mean.append(sensitivity_mean)
            specificities_mean.append(specificity_mean)

            print('MEDIAN', sensitivity_median, specificity_median)
            
            save_metrics_for_threshold(npy_folder, threshold)

            with open(os.path.join(npy_folder, 'metrics_threshold_relation_by_patient.csv'), 'a', newline='') as csvfile:
                fieldnames = ['time', 'threshold', 'sensitivity_median', 'specificity_median','sensitivity_mean', 'specificity_mean', 'dice_median', 'dice_mean']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold':str(threshold),
                                 'sensitivity_median':str(sensitivity_median),
                                 'specificity_median':str(specificity_median),
                                 'sensitivity_mean':str(sensitivity_mean),
                                 'specificity_mean':str(specificity_mean),
                                 'dice_median':str(np.nanmedian(dices)),
                                'dice_mean':str(np.nanmean(dices))})

        else:
            fpr, tpr, thresholds_p = metrics.roc_curve(np.rint(all_gt), all_predictions_raw)
            threshold_p = thresholds_p[np.argmax(tpr - fpr)]
            roc_auc = metrics.auc(fpr, tpr)
            predictions = np.where(all_predictions_raw > threshold, 1, 0)


            sensitivity, specificity, dice = test.Tester.count_metrics(np.rint(all_gt), predictions, str(threshold), save_stats=False, return_dice=True)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

            print('ALL sens, spec, dice, auc, threshold', sensitivity, specificity, dice, roc_auc, threshold_p)

            with open(os.path.join(npy_folder, 'metrics_threshold_relation_all.csv'), 'a', newline='') as csvfile:
                fieldnames = ['time', 'threshold', 'sensitivity', 'specificity']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), #TODO. если буду это использавать, то не забыть добавить median, mean, dice
                                 'threshold':str(threshold),
                                 'sensitivity':str(sensitivity),
                                 'specificity':str(specificity)})

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




def count_ROC(csv_path, save_path, checkpoint=None):
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

            #tester = test.Tester('cp-00'+str(config.EPOCHS), ['data'], '', MODEL_FOLDER=row[5])


            sensitivity = specificity = 0
            tester = None
            #!!!!!!!!!!!!!!!!!!!!!!!!REMOVE
            checkpoint_number = int(checkpoint.split('-')[-1])
            print(checkpoint_number)
            '''if int(row[1]) in [0, 1, 2, 3] and not checkpoint_number in [100, 200]:
                tester1 = test.Tester( 'cp-0'+str(checkpoint_number - 50), ['data'], '', MODEL_FOLDER=row[5])

                sensitivity1, specificity1 = tester1.test_one_image(row[4],
                                                                 path_image=row[4] + '_Mask JW Kolo.png',
                                                                 save=False,
                                                                 show=False,
                                                                 test_all_spectra=False,
                                                                 save_stats=False,
                                                                 folder_name='')

                tester2 = test.Tester( 'cp-0'+str(checkpoint_number + 50), ['data'], '', MODEL_FOLDER=row[5])

                sensitivity2, specificity2 = tester2.test_one_image(row[4],
                                                                   path_image=row[4] + '_Mask JW Kolo.png',
                                                                   save=False,
                                                                   show=False,
                                                                   test_all_spectra=False,
                                                                   save_stats=False,
                                                                   folder_name='')

                specificity = np.mean([specificity1, specificity2])
                sensitivity = np.mean([sensitivity1, sensitivity2])

                tester = tester1
                tester.all_predictions_raw = list(np.mean([tester1.all_predictions_raw, tester2.all_predictions_raw], axis=0))
                tester.all_gt = tester1.all_gt
            else:'''
            tester = test.Tester( checkpoint, ['data'], '', MODEL_FOLDER=row[5])

            sensitivity, specificity = tester.test_one_image(row[4],
                                  path_image=row[4] + '_Mask JW Kolo.png',
                                  save=False,
                                  show=False,
                                  test_all_spectra=False,
                                  save_stats=False,
                                  folder_name='', 
                                  spectrum_shift=-1)

            print('Sensitivity on ', checkpoint, '- ', sensitivity, ';was - ', row[2], ' diff - ', sensitivity - float(row[2]))

            all_predictions_raw += tester.all_predictions_raw
            all_gt += tester.all_gt
            predictions_by_patient.append(tester.all_predictions_raw)
            gt_by_patient.append(tester.all_gt)

    print(np.array(predictions_by_patient).shape, np.array(gt_by_patient).shape)

    #save for future part
    np.save(os.path.join(save_path, 'all_predictions_raw.npy'),  all_predictions_raw)
    np.save(os.path.join(save_path, 'predictions_by_patient.npy'),  predictions_by_patient)
    np.save(os.path.join(save_path, 'all_gt.npy'), all_gt)
    np.save(os.path.join(save_path, 'gt_by_patient.npy'), gt_by_patient)

    #sensitivity, specificity part
    sensitivity, specificity = test.Tester.count_metrics(all_gt, np.rint(np.array(all_predictions_raw)), 'vvv', save_stats=False)
    print('Complete sensitivity, specificity:', sensitivity, specificity)

    #roc auc part
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

    del fpr, tpr, all_gt, all_predictions_raw, predictions_by_patient, gt_by_patient







def run_csv_and_save_images(csv_path, save_path, test_all_spectra=True):
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

            tester.test_one_image(row[4],
                                  path_image=row[4] + '_Mask JW Kolo.png',
                                  save=True,
                                  show=False,
                                  test_all_spectra=test_all_spectra,
                                  save_stats=False,
                                  folder_name='')

            #test_one_image(self, path_dat, path_image=None, save=False, show=True, test_all_spectra = False, save_stats = False, folder_name = '')



def dropout_experiment():
    for dropout in tqdm(np.linspace(0, 1, 10)):
        config.DROPOUT_VALUE = dropout
        old_model_name = config.MODEL_NAME
        config.MODEL_NAME += '_dropout_' + str(round(config.DROPOUT_VALUE, 2))

        train()

        config.MODEL_NAME = old_model_name

#name - of experiment, name of subfolder that will be created in "test"
#paths - paths with models that would be tested
def test_experiment(name, paths):
    SAVING_PATH = os.path.join('test', name)
    if not os.path.exists(SAVING_PATH):
        os.mkdir(SAVING_PATH)

    for path in paths:
        tester = test.Tester('logs', path.split('\\')[-1], 'cp-00'+str(config.EPOCHS), ['data', 'test_test'], SAVING_PATH)
        tester.test_ALL_images(test_all_spectra=False, save=True, show=False, save_stats=True)

#(3737608, 94) (383654, 94)

def cross_validation(root_folder_name):
    config.MODEL_NAME_PATHS.append(root_folder_name)

    root_folder = os.path.join(*config.MODEL_NAME_PATHS)
    config.MODEL_NAME = config.get_model_name(config.MODEL_NAME_PATHS)
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)

    '''if root_folder != '':
        config.MODEL_NAME_PATHS.append(root_folder)
        config.MODEL_NAME = os.path.join(root_folder, config.MODEL_NAME.split('/')[-1])
    else:
        config.MODEL_NAME = os.path.join(config.MODEL_NAME, config.MODEL_NAME.split('/')[-1])'''

    paths = []
    for data_path in config.DATA_PATHS:
        paths += glob.glob(os.path.join(data_path, '*.dat'))

    csv_filename = os.path.join(root_folder, root_folder_name + '_stats'+ datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S") +'.csv')

    splits = np.array_split(range(len(paths)), config.CROSS_VALIDATION_SPLIT)

    for ind, indexes in enumerate(splits):
        old_model_name = config.MODEL_NAME
        if len(indexes) > 1:
            for i in indexes:
                config.MODEL_NAME += '_' + str(i)
        else:
            config.MODEL_NAME += '_' + str(ind) + '_' + np.array(paths)[indexes][0].split("/")[-1].split(".")[0].split('SpecCube')[0]
        
        model = train(paths=paths, except_indexes=[p.split("/")[-1].split(".")[0].split('SpecCube')[0] for p in np.array(paths)[indexes]])
        #model = train(paths=paths, except_indexes=indexes) #for old CV when I used as indexes numbers except names 
        #for i, path in enumerate(paths):  # full cross_valid
        for i in indexes:
        #for i, path in zip(range(7, len(paths)), paths[7:]): #cross valid from several index


            #CHECKPOINT, TEST_PATHS, SAVING_PATH, LOGS_PATH='', MODEL_NAME='', MODEL_FOLDER=''):
            tester = test.Tester( f'cp-{config.EPOCHS:04d}', ['data'], '', LOGS_PATH=root_folder, MODEL_NAME=config.MODEL_NAME.split('\\')[-1])

            sensitivity, specificity = tester.test_one_image(paths[i],
                                path_image=paths[i] + '_Mask JW Kolo.png',
                                save=False,
                                show=False,
                                test_all_spectra=False,
                                save_stats=False,
                                folder_name=config.MODEL_NAME,
                                spectrum_shift=-1)   #be carefull!

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

def compare_checkpoints():
    #rg = np.linspace(2, 30, 15).astype(int)
    #rg = np.linspace(2, 20, 10).astype(int)
    rg = np.linspace(40, 200, 5).astype(int)
    checkpoints = [f'cp-{i:04d}' for i in rg]
    print(checkpoints)
    #f'cp-{config.EPOCHS:04d}'
    #save_path_ = 'test/CV_combi_WRA_50max_8inc_30epochs_1pat'
    #save_path_ = 'test/CV_combi_WRA_fixed_preproc_smaller_model_50max_20epochs_1pat'
    save_path_ = 'test/CV_aug'
    for checkpoint in checkpoints:
        print(checkpoint)
        
        if not os.path.exists(save_path_):
            os.mkdir(save_path_)
        
        save_path = os.path.join(save_path_, checkpoint)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        #count_ROC('logs/CV_combi_WRA_50max_8inc_30epochs_1pat/CV_combi_WRA_50max_8inc_30epochs_1pat_stats_25.08.2021-12_16_30.csv', save_path, checkpoint=checkpoint)
        #count_ROC('logs/CV_combi_WRA_fixed_preproc_smaller_model_50max_20epochs_1pat/CV_combi_WRA_fixed_preproc_smaller_model_50max_20epochs_1pat_stats_24.08.2021-18_46_19.csv', save_path, checkpoint=checkpoint)
        count_ROC('logs/CV_aug/CV_aug_stats_16.08.2021-11_58_29.csv', save_path, checkpoint=checkpoint)

        count_metrics_on_diff_thresholds(save_path, all=True, threshold_range_params=[0.05, 0.95, 19])

if __name__ =='__main__':
    
    try:
        compare_checkpoints()

        #count_ROC('logs/inception_l2_norm/inception_l2_norm_stats_16.01.2021-08_56_39.csv', 'save_path', checkpoint='cp-0250')

        #save_metrics_for_threshold('test/inception_l2_norm/cp-0250', 0.45)

        #count_metrics_on_diff_thresholds('test/inception_l2_norm_all_data/cp-0100', all=False, threshold_range_plain=[0.47])

        '''count_ROC('logs\\inception_cross_validation\\inception_cross_validation_stats_06.01.2021-11_03_50.csv', 'test/inception_cv_images')

        npy_folder = 'test/inception_cv_images'
        all_gt = np.loadtxt(os.path.join(npy_folder, 'all_gt.txt'))
        all_predictions_raw = np.loadtxt(os.path.join(npy_folder, 'all_predictions_raw.txt'))

        print(all_gt[:10])

        sensitivity, specificity = test.Tester.count_metrics(all_gt, np.rint(np.array(all_predictions_raw)), 'vvv', save_stats=False)
        print('Complete sensitivity, specificity:', sensitivity, specificity)'''

        #run...save_path='test/inception_cv_images/not_all_spectra'
        #cross_validation('CV_combi_WRA_50max_8inc_30epochs_1pat')

        #paths = glob.glob('logs/test_inception*')
        #test_experiment('dropout_experiment', paths)

        #run_csv_and_save_images('logs\\inception_l2_norm\\inception_l2_norm_stats_16.01.2021-08_56_39.csv', 'test/inception_l2_norm/all_spectrum', test_all_spectra=True)
    except Exception as e:
        print(e)
        
        if config.TELEGRAM_SENDING:
            send_tg_message(f'Mariia, ERROR!!!, In CV error {e}')
        
        raise e 