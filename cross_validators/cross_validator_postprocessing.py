"""
Cross-validation with post-processing, which provides serious improvement of performance, excpecially if model classify pixel by pixel.
Paper about post-processing: https://www.mdpi.com/2072-6694/15/7/2157
Documentation on wiki: https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/Post-processing
After inizialization post-processing starts form evaluation() function
"""

import os
from glob import glob
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from tqdm import tqdm
import csv

from data_utils.data_loaders.data_loader_base import DataLoader
from data_utils.preprocessor import Preprocessor
from cross_validators.cross_validator_base import CrossValidatorBase
import config
from evaluation.validator import Validator
from provider import get_whole_analog_of_data_loader, get_evaluation


class CrossValidatorPostProcessing(CrossValidatorBase):
    def __init__(self, name, cross_validation_type='algorithm_with_threshold', **kwargs):
        super().__init__(name)
        self.LABELED_NPZ_FOLDER = config.RAW_NPZ_PATH
        self.cross_validation_type = cross_validation_type
        self.whole_database = get_whole_analog_of_data_loader(config.DATABASE)
        self.original_database = config.DATABASE

        self.WHOLE_CUBES_FOLDER = os.path.join(self.LABELED_NPZ_FOLDER, 'whole')
        if not os.path.exists(self.WHOLE_CUBES_FOLDER):
            os.mkdir(self.WHOLE_CUBES_FOLDER)

        print('WHOLE_CUBES_PATH', self.WHOLE_CUBES_FOLDER)
        self.WHOLE_CUBES = glob(os.path.join(self.WHOLE_CUBES_FOLDER, '*.npz'))
        print('WHOLE_CUBES', self.WHOLE_CUBES)
        self.LABELED_CUBES = glob(os.path.join(self.LABELED_NPZ_FOLDER, '*.npz'))
        print('LABELED_CUBES', self.LABELED_CUBES)

        initialize_execution_flags(**kwargs)

        print('execution_flags:', self.execution_flags)

        search_folder = os.path.join(self.project_folder, 'logs', name)
        print(search_folder)
        self.training_csv_path = CrossValidatorBase.get_csv(search_folder)

        self.saving_folder = os.path.join(self.project_folder, 'test', name)
        if not os.path.exists(self.saving_folder):
            os.mkdir(self.saving_folder)

        self.saving_folder_with_checkpoint = os.path.join(self.saving_folder, 'cp-0000')
        if not os.path.exists(self.saving_folder_with_checkpoint):
            os.mkdir(self.saving_folder_with_checkpoint)

        self.evaluator = get_evaluation(name)

        self.whole_predictions_filename = 'predictions_whole.npy'
        self.predictions_filename = self.evaluator.predictions_npy_filename
        self.file_with_postprocessed_predictions = 'predictions_postprocessed.npy'
        self.file_with_postprocessed_predictions_only_for_labeled_samples = 'predictions_postprocessed_only_labeled_samples.npy'

    def initialize_execution_flags(self, **kwargs):
        if not kwargs:
            self.execution_flags = {
                "generate_whole_cubes": False,
                # by default if "whole" folder is empty than we generate whole cubes, otherwise we don't. But with generate_whole_cubes it's possible to forse generate
                "get_predictions_for_whole_cubes": False,
                # by default if there is no predictions_whole.npy in test/name/cp-0000 than we count predoctions for whole cubes, otherwise - we don't. But with get_predictions_for_whole_cubes it's possible to forse count
                "save_predictions_and_metrics_on_labeled": {
                    # for detailed documentation of params in this dictionary see documentation for evaluation/evaluation_base.py/EvaluationBase.save_predictions_and_metrics()
                    "save_predictions": False,
                    "metrics": {
                        'save_metrics': False,
                        'checkpoints_range': None,
                        'checkpoints_raw_list': None,
                        'thresholds_range': None,
                        'thresholds_raw_list': None,
                        'save_curves': False
                    }
                },
                "check": {  # what thresholds and median filter sizes to check
                    "median_filters_raw_list": None,    # you can write MF size as list, for example, [5, 10, 15, 20]
                    "median_filters_range": None,       # or like range [5, 20, 4] (divide space between 5 and 20 in 4 numbers)
                    "thresholds_raw_list": None,        # the same system for thresholds
                    "thresholds_range": None
                }
            }
        else:
            self.execution_flags = kwargs['execution_flags']
    
    def evaluation(self, **kwargs):    # entry point

        self.generate_whole_cubes()
        self.calculate_predictions_on_whole_cubes()
        self.calculate_predictions_only_on_labeled_samples()

        if self.cross_validation_type == 'algorithm_with_threshold':
            thresholds_range = self.execution_flags['check']['thresholds_range']
            thresholds_raw_list = self.execution_flags['check']['thresholds_raw_list']
        if self.cross_validation_type == 'algorithm_plain':
            thresholds_raw_list = [-1]
            thresholds_range = None

        self.check_different_MF_sizes_and_thresholds(
            thresholds_range=thresholds_range,
            thresholds_raw_list=thresholds_raw_list,
            median_filters_raw_list=self.execution_flags['check']['median_filters_raw_list'],
            median_filters_range=self.execution_flags['check']['median_filters_range'])


    def generate_whole_cubes(self):
        if len(self.WHOLE_CUBES) != len(self.LABELED_CUBES) or self.execution_flags['generate_whole_cubes']:
            print('---------- Cubes generation is started------------')
            execution_flags = Preprocessor.get_execution_flags_for_pipeline_with_all_true()
            execution_flags['load_data_with_dataloader'] = True
            execution_flags['add_sample_weights'] = False
            execution_flags['scale'] = True
            execution_flags['shuffle'] = False

            config.DATABASE = self.whole_database

            preprocessor = Preprocessor()
            preprocessor.pipeline(config.RAW_SOURCE_PATH, self.WHOLE_CUBES_FOLDER,
                                  execution_flags=execution_flags)
            config.DATABASE = self.original_database

            print('---------- Cubes generation is finished------------')
        else:
            print("!!!---------- We are not generating whole cubes, because they have already existed and 'generate_whole_cubes' is set to False ----------!!!")

    def calculate_predictions_on_whole_cubes(self):
        predictions_npz_exists = os.path.exists(os.path.join(self.saving_folder_with_checkpoint,
                                                             self.whole_predictions_filename))
        if not predictions_npz_exists or self.execution_flags['get_predictions_for_whole_cubes']:
            print('---------- Calculation of predictions on whole cubes is started------------')
            config.USE_ALL_LABELS = True

            self.evaluator.save_predictions_and_metrics(
                training_csv_path=self.training_csv_path,
                save_predictions=True,
                npz_folder=self.WHOLE_CUBES_FOLDER,
                save_curves=False,
                predictions_npy_filename=self.whole_predictions_filename,
                checkpoints_range=self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics'][
                    'checkpoints_range'],
                checkpoints_raw_list=self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics'][
                    'checkpoints_raw_list']
            )
            config.USE_ALL_LABELS = False
            print('---------- Calculation of predictions on whole cubes is finished ------------')
        else:
            print("!!!---------- We don't calculate predictions for whole cubes ----------!!!")

    def calculate_predictions_only_on_labeled_samples(self):
        predictions_exist = os.path.exists(os.path.join(self.saving_folder_with_checkpoint, self.predictions_filename))

        if not predictions_exist or self.execution_flags['save_predictions_and_metrics_on_labeled'][
            'save_predictions'] or \
                self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics']['save_metrics']:
            print('---------- Calculation of predictions on labeled samples is started ------------')
            thresholds_range = self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics'][
                    'thresholds_range']
            thresholds_raw_list = self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics'][
                    'thresholds_raw_list']

            save_predictions = False
            if not predictions_exist or self.execution_flags['save_predictions_and_metrics_on_labeled'][
                'save_predictions']:
                save_predictions = True

            self.evaluator.save_predictions_and_metrics(
                training_csv_path=self.training_csv_path,
                npz_folder=self.LABELED_NPZ_FOLDER,
                predictions_npy_filename=self.predictions_filename,

                checkpoints_range=self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics'][
                    'checkpoints_range'],
                checkpoints_raw_list=self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics'][
                    'checkpoints_raw_list'],
                thresholds_range=thresholds_range,
                thresholds_raw_list=thresholds_raw_list,
                save_predictions=save_predictions,
                save_curves=self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics']['save_curves']
            )
            print('---------- Calculation of predictions on labeled samples is finished ------------')
        else:
            print("!!!---------- We don't calculate predictions for labeled samples ----------!!!")
        return

    @staticmethod
    def get_MF_sizes_and_thresholds(median_filters_range, median_filters_raw_list, thresholds_range,
                                          thresholds_raw_list):
        if median_filters_range is not None and median_filters_raw_list is not None:
            raise ValueError("Error! Both median_filters_range and median_filters_raw_list are specified. Please "
                             "specify only one of them")
        if thresholds_range is not None and thresholds_raw_list is not None:
            raise ValueError("Error! Both median_filters_range and median_filters_raw_list are specified. Please "
                             "specify only one of them")
        if thresholds_range is None and thresholds_raw_list is None:
            raise ValueError("Error! Both median_filters_range and median_filters_raw_list are not specified. Please "
                             "specify one of them")
        median_filters = []
        if median_filters_range is not None:
            median_filters = np.round(np.linspace(median_filters_range[0], median_filters_range[1],
                                                  median_filters_range[2]), 4)
        if median_filters_raw_list is not None:
            median_filters = median_filters_raw_list.copy()

        thresholds = []
        if thresholds_range is not None:
            thresholds = np.round(np.linspace(thresholds_range[0], thresholds_range[1], thresholds_range[2]), 4)
        if thresholds_raw_list is not None:
            thresholds = thresholds_raw_list.copy()
        return median_filters, thresholds

    def check_different_MF_sizes_and_thresholds(self, thresholds_range=None, thresholds_raw_list=None,
                                                    median_filters_range=None, median_filters_raw_list=None):
        data = np.load(
            os.path.join(self.saving_folder_with_checkpoint, self.whole_predictions_filename), allow_pickle=True)

        median_filters, thresholds = CrossValidatorPostProcessing.get_MF_sizes_and_thresholds(
            median_filters_range,
            median_filters_raw_list,
            thresholds_range,
            thresholds_raw_list)
        print(
            f'---------- We are starting to check different median filter sizes {median_filters} and thresholds {thresholds} ----------')
        original_filename = self.evaluator.comparable_characteristics_csvname
        original_metrics_filename = self.evaluator.metrics_filename_base

        for med_filter in median_filters:
            for threshold in thresholds:
                print(f'Check median filter size - {med_filter}, and threshold -{threshold}')
                folder_name = f"mf_{med_filter}_t_{threshold}"
                folder = os.path.join(self.saving_folder_with_checkpoint, folder_name)
                if not os.path.exists(folder):
                    os.mkdir(folder)

                postprocessed_predictions = {}

                for patient in data:
                    predictions_postprocessed = self.median_filter(patient, threshold, med_filter, folder)
                    postprocessed_predictions.update({
                        patient['name']: {
                            'predictions': predictions_postprocessed,
                            'gt': patient['gt'],
                        }
                    })

                np.save(os.path.join(folder, self.file_with_postprocessed_predictions), postprocessed_predictions)
                self.save_labeled_samples_from_postprocessed_whole_cubes(folder)
                
                self.evaluator.comparable_characteristics_csvname = "compare_all_thresholds_postprocessed.csv"
                if self.cross_validation_type == 'algorithm_plain':
                    self.evaluator.comparable_characteristics_csvname = "compare_all_thresholds_postprocessed_AP.csv"
                    
                self.evaluator.metrics_filename_base += '_postprocessed_' + str(med_filter)
                self.evaluator.metrics_filename_base = folder + config.SYSTEM_PATHS_DELIMITER \
                                                       + self.evaluator.metrics_filename_base
                self.evaluator.additional_columns = {'median': med_filter}

                thresholds_raw_list = [threshold]
                thresholds_range = None
                if self.cross_validation_type == 'algorithm_plain':
                    if self.execution_flags['check']['thresholds_range'] is not None:
                        thresholds_range = [self.execution_flags['check']['thresholds_range']]
                    thresholds_raw_list = self.execution_flags['check']['thresholds_raw_list']
                print('thresholds_raw_list, thresholds_range', thresholds_raw_list, thresholds_range)

                self.evaluator.save_predictions_and_metrics(
                    training_csv_path=self.training_csv_path,
                    save_predictions=False,
                    npz_folder=self.LABELED_NPZ_FOLDER,
                    save_curves=False,
                    predictions_npy_filename=folder_name + config.SYSTEM_PATHS_DELIMITER + self.file_with_postprocessed_predictions_only_for_labeled_samples,
                    thresholds_raw_list=thresholds_raw_list,
                    thresholds_range=thresholds_range,
                    checkpoints_range=self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics'][
                        'checkpoints_range'],
                    checkpoints_raw_list=self.execution_flags['save_predictions_and_metrics_on_labeled']['metrics'][
                        'checkpoints_raw_list']
                )

                self.evaluator.metrics_filename_base = original_metrics_filename
        self.evaluator.comparable_characteristics_csvname = original_filename
        self.evaluator.additional_columns = {}

        print(f'---------- Checking different median filter sizes and thresholds is finished ----------')

    def median_filter(self, patient, threshold, median_filter_size, folder):

        size = patient['size']
        predictions = np.reshape(np.array(patient['predictions'])[:, 0], size)

        if threshold != -1:
            predictions[predictions >= threshold] = 1
            predictions[predictions < threshold] = 0

        predictions_postprocessed = median_filter(predictions, size=median_filter_size)
        
        '''postprocessed_predictions.append(np.reshape(predictions_postprocessed, size))
        
        gt = np.reshape(np.array(patient['gt']), size).astype(np.float)
        gt_ = np.array(gt)
        gt[gt_ == 2.] = 0.
        gt[gt_ == 0.] = 0.5

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=200)
        ax1.imshow(predictions)
        ax2.imshow(predictions_postprocessed)
        ax3.imshow(gt, vmin=0, vmax=1)
        ax3.set_title('Ground Truth. \n Yellow - cancer, \n blue - healthy, \n hell blue - background',
                      fontdict={'fontsize': 6})
        ax1.set_title('Predictions from \n the network. \n Yellow - cancer, \nblue - healthy', fontdict={'fontsize': 6})
        ax2.set_title('Predictions after \n median filter', fontdict={'fontsize': 6})
        plt.savefig(os.path.join(folder, str(patient['name']) + '.png'))
        plt.clf()
        plt.cla()
        plt.close(fig)'''

        return predictions_postprocessed

    def save_labeled_samples_from_postprocessed_whole_cubes(self, folder):
        postprocessed_predictions = np.load(os.path.join(folder,
                                                    self.file_with_postprocessed_predictions), allow_pickle=True).item()

        result = []

        with open(self.training_csv_path, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                name = DataLoader.get_name_easy(row[4], delimiter='/')
                data = np.load(os.path.join(config.RAW_NPZ_PATH, name + '.npz'))
                indexes_in_datacube = data['indexes_in_datacube']
                predictions = postprocessed_predictions[name]['predictions']
                predictions = predictions[indexes_in_datacube[:, 0], indexes_in_datacube[:, 1]]

                gt = data['y']
                indx_ = np.zeros(gt.shape).astype(bool)
                for label in config.LABELS_OF_CLASSES_TO_TRAIN:
                    indx_ = indx_ | (gt == label)
                gt = gt[indx_]
                predictions = predictions[indx_]

                result.append({'name': name,
                               'gt': gt,
                               'predictions': predictions})

        np.save(os.path.join(folder, self.file_with_postprocessed_predictions_only_for_labeled_samples), result)
