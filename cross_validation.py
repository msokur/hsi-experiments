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

    with open(os.path.join(npy_folder, 'metrics_by_threshold_'+str(threshold)+'.csv'), 'a', newline='') as csvfile:
        fieldnames = ['time', 'threshold', 'sensitivity', 'specificity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for patient in range(predictions_by_patient.shape[0]):
            predictions = np.where(np.array(predictions_by_patient[patient]) > threshold, 1, 0)

            sensitivity_p, specificity_p= test.Tester.count_metrics(np.rint(gt_by_patient[patient]), predictions, str(threshold), save_stats=False)
            sensitivities.append(sensitivity_p)
            specificities.append(specificity_p)

            writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                             'threshold':str(threshold),
                             'sensitivity':str(sensitivity_p),
                             'specificity':str(specificity_p)})

        sensitivity = np.median(sensitivities)
        specificity = np.median(specificities)

        writer.writerow({'time':"GESAMT",
                         'threshold':str(threshold),
                         'sensitivity':str(sensitivity),
                         'specificity':str(specificity)})

        writer.writerow({'time':"STD",
                         'threshold':str(threshold),
                         'sensitivity':str(np.std(sensitivities)),
                         'specificity':str(np.std(specificities))})




def count_metrics_on_diff_thresholds(npy_folder, all=False):
    all_gt = np.load(os.path.join(npy_folder, 'all_gt.npy'))
    predictions_by_patient = np.load(os.path.join(npy_folder, 'predictions_by_patient.npy'), allow_pickle=True)
    all_predictions_raw = np.load(os.path.join(npy_folder, 'all_predictions_raw.npy'))
    gt_by_patient = np.load(os.path.join(npy_folder, 'gt_by_patient.npy'), allow_pickle=True)

    sensitivities = []
    specificities = []

    rng = np.linspace(0, 1, 21)

    for threshold in tqdm(rng):
        if all:
            sens = []
            spec = []
            for patient in range(predictions_by_patient.shape[0]):
                predictions = np.where(predictions_by_patient[patient] > threshold, 1, 0)

                sensitivity_p, specificity_p= test.Tester.count_metrics(np.rint(gt_by_patient[patient]), predictions, str(threshold), save_stats=False)
                sens.append(sensitivity_p)
                spec.append(specificity_p)

            sensitivity = np.median(sens)
            specificity = np.median(spec)

            sensitivities.append(sensitivity)
            specificities.append(specificity)

            print('MEDIAN', sensitivity, specificity)

            with open(os.path.join(npy_folder, 'metrics_threshold_relation_median_by_patient.csv'), 'a', newline='') as csvfile:
                fieldnames = ['time', 'threshold', 'sensitivity', 'specificity']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold':str(threshold),
                                 'sensitivity':str(sensitivity),
                                 'specificity':str(specificity)})

        else:
            predictions = np.where(all_predictions_raw > threshold, 1, 0)

            sensitivity, specificity = test.Tester.count_metrics(np.rint(all_gt), predictions, str(threshold), save_stats=False)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

            with open(os.path.join(npy_folder, 'metrics_threshold_relation_all.csv'), 'a', newline='') as csvfile:
                fieldnames = ['time', 'threshold', 'sensitivity', 'specificity']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'time':datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'threshold':str(threshold),
                                 'sensitivity':str(sensitivity),
                                 'specificity':str(specificity)})

    plt.plot(rng, sensitivities)
    plt.plot(rng, specificities)
    plt.savefig(os.path.join(npy_folder, 'thresholds_metrics_curves_median.png'))
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
                                  folder_name='')

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
    print('threshold', threshold)
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

    paths = glob.glob(os.path.join(config.DATA_PATHS[0], '*.dat'))

    csv_filename = os.path.join(root_folder, root_folder_name + '_stats'+ datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S") +'.csv')

    splits = np.array_split(range(len(paths)), config.CROSS_VALIDATION_SPLIT)

    for indexes in splits:
        old_model_name = config.MODEL_NAME
        if len(indexes) > 1:
            for i in indexes:
                config.MODEL_NAME += '_' + str(i)
        else:
            config.MODEL_NAME += '_' + str(i) + '_' + paths[i].split('\\')[-1].split('.')[0]

        model = train(paths=paths, except_indexes=indexes)
        #for i, path in enumerate(paths):  # full cross_valid
        for i in indexes:
        #for i, path in zip(range(7, len(paths)), paths[7:]): #cross valid from several index


            #model = train(paths=paths, except_indexes=[i])


            #test.model = model

            #CHECKPOINT, TEST_PATHS, SAVING_PATH, LOGS_PATH='', MODEL_NAME='', MODEL_FOLDER=''):
            tester = test.Tester( f'cp-{config.EPOCHS:04d}', ['data'], '', LOGS_PATH=root_folder, MODEL_NAME=config.MODEL_NAME.split('\\')[-1])

            #test_one_image(self, path_dat, path_image=None, save=False, show=True, test_all_spectra = False, save_stats = False, folder_name = '')

            sensitivity, specificity = tester.test_one_image(paths[i],
                                path_image=paths[i] + '_Mask JW Kolo.png',
                                save=False,
                                show=False,
                                test_all_spectra=False,
                                save_stats=False,
                                folder_name=config.MODEL_NAME)

            print('For path=', paths[i], ', (index: ', str(i), ') sensitivity= ', str(sensitivity), ' specificity= ', str(specificity), ' MODEL_NAME = ', config.MODEL_NAME)

            with open(csv_filename,'a', newline='') as csvfile: # for full cross_valid and for separate file

            #with open(r'F:\\HSI_data\\logs\\inception_cross_validation\\inception_cross_validation_stats_06.01.2021-11_03_50.csv','a', newline='') as csvfile:
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
    rg = np.linspace(75, 200, 6).astype(int)
    checkpoints = [f'cp-{i:04d}' for i in rg]
    print(checkpoints)
    #f'cp-{config.EPOCHS:04d}'
    save_path_ = 'test/lstm_inception_8'
    for checkpoint in checkpoints[-2:]:
        save_path = os.path.join(save_path_, checkpoint)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        count_ROC('logs/lstm_inception_8/lstm_inception_8_stats_12.02.2021-16_49_31.csv', save_path, checkpoint=checkpoint)

        count_metrics_on_diff_thresholds(save_path, all=True)

if __name__ =='__main__':

    compare_checkpoints()
    #save_metrics_for_threshold('test/inception_l2_norm/cp-0250', 0.45)

    #count_metrics_on_diff_thresholds('test/lstm_inception_8/cp-0075')

    '''count_ROC('logs\\inception_cross_validation\\inception_cross_validation_stats_06.01.2021-11_03_50.csv', 'test/inception_cv_images')

    npy_folder = 'test/inception_cv_images'
    all_gt = np.loadtxt(os.path.join(npy_folder, 'all_gt.txt'))
    all_predictions_raw = np.loadtxt(os.path.join(npy_folder, 'all_predictions_raw.txt'))

    print(all_gt[:10])

    sensitivity, specificity = test.Tester.count_metrics(all_gt, np.rint(np.array(all_predictions_raw)), 'vvv', save_stats=False)
    print('Complete sensitivity, specificity:', sensitivity, specificity)'''

    #run...save_path='test/inception_cv_images/not_all_spectra'
    #cross_validation('lstm_inception_8')

    #paths = glob.glob('logs/test_inception*')
    #test_experiment('dropout_experiment', paths)

    #run_csv_and_save_images('logs\\inception_l2_norm\\inception_l2_norm_stats_16.01.2021-08_56_39.csv', 'test/inception_l2_norm/all_spectrum', test_all_spectra=True)