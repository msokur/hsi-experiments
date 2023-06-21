"""
Cross-validation with post-processing, which provides serious improvement of performance, excpecially if model classify pixel by pixel.
Paper about post-processing: https://www.mdpi.com/2072-6694/15/7/2157
Documentation on wiki: https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/Post-processing

You can start post-processing with postprocessing_for_one_model() funtion in cross_validation.py
After initialization of CrossValidatorPostProcessing execution goes to CrossValidatorPostProcessing.evaluation() function
MF - Median Filter
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

from data_utils.data_loaders.archive.data_loader_base import DataLoader
from data_utils.archive.preprocessor import Preprocessor
from cross_validators.cross_validator_base import CrossValidatorBase
import config
from provider import get_whole_analog_of_data_loader, get_evaluation


class CrossValidatorPostProcessing(CrossValidatorBase):
    # the other option for cross_validation_type is 'algorithm_with_threshold', but as pointed in documentation we reccomend to use 'algorithm_plain' 
    def __init__(self, name, cross_validation_type='algorithm_plain', **kwargs):
        if cross_validation_type == 'algorithm_with_threshold' and len(config.LABELS_OF_CLASSES_TO_TRAIN) > 2:
            raise ValueError('Error! AWT (algorithm_with_threshold) could only be used with binary classification. Please use "algorithm_plain" for multiclass classification')
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

        self.set_configuration(**kwargs)

        print('Post-processing configuration:', self.configuration)

        search_folder = os.path.join(self.project_folder, 'logs', name)
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
        self.file_with_postprocessed_predictions_for_labeled_samples = 'predictions_postprocessed_for_labeled_samples.npy'

    def evaluation(self, **kwargs):
        """
            Steps - 1 variant (median filter is applied to predictions 0-1):
            1. Generate whole cubes, if needed
            2. Make predictions for this whole cubes, if needed
            3. Make save_ROC_... for labeled and for thresholds, if needed
            4. Get the best threshold
            5. Due to this threshold create predictions map 0-1
            6. Apply median filter
            7. Get labeled with indexes_in_cube
            8. Count evaluation again for the best threshold

            Steps - 2 varian (median filter is applied to raw predictions)
            1.-2. are the same
            3. Apply median filter on the raw predictions
            4. Get labeled with indexes_in_cube
            5. Make save_ROC...

        """

        self.generate_whole_cubes()
        self.calculate_predictions_on_whole_cubes()
        self.calculate_predictions_and_metrics_on_labeled_samples()

        if self.cross_validation_type == 'algorithm_with_threshold':
            thresholds = self.configuration['postprocessing']['thresholds']
        if self.cross_validation_type == 'algorithm_plain':
            thresholds = [-1] # we fill thresholds with -1 to skip thresholding in CrossValidatorPostProcessing.median_filter() function

        self.postprocessing(thresholds=thresholds, MF_sizes=self.configuration['postprocessing']['MF_sizes'])


    def generate_whole_cubes(self):
        if len(self.WHOLE_CUBES) != len(self.LABELED_CUBES) or self.configuration['generate_whole_cubes']:
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
        if not predictions_npz_exists or self.configuration['calculate_predictions_for_whole_cubes']:
            print('---------- Calculation of predictions on whole cubes is started------------')
            config.USE_ALL_LABELS = True

            self.evaluator.calculate_and_save_predictions(
                training_csv_path=self.training_csv_path,
                npz_folder=self.WHOLE_CUBES_FOLDER,
                predictions_npy_filename=self.whole_predictions_filename
            )
            config.USE_ALL_LABELS = False
            print('---------- Calculation of predictions on whole cubes is finished ------------')
        else:
            print("!!!---------- We don't calculate predictions for whole cubes ----------!!!")

    def calculate_predictions_and_metrics_on_labeled_samples(self):
        predictions_exist = os.path.exists(os.path.join(self.saving_folder_with_checkpoint, self.predictions_filename))

        if not predictions_exist or self.configuration['save_predictions_and_evaluate_on_labeled_samples']['save_predictions']:
            print('---------- Calculation of predictions on labeled samples is started ------------')
            self.evaluator.calculate_and_save_predictions(training_csv_path=self.training_csv_path,
                                                        npz_folder=self.LABELED_NPZ_FOLDER,
                                                        predictions_npy_filename=self.predictions_filename,
                                                        checkpoints=self.configuration['save_predictions_and_evaluate_on_labeled_samples']['metrics']['checkpoints'])
        
                
            print('---------- Calculation of predictions on labeled samples is finished ------------')
        else:
            print("!!!---------- We don't calculate predictions on labeled samples ----------!!!")

            
        if self.configuration['save_predictions_and_evaluate_on_labeled_samples']['metrics']['save_metrics']:   
            print('---------- Calculation of metrics on labeled predictions is started ------------')
            self.evaluator.evaluate(predictions_npy_filename=self.predictions_filename,
                                    checkpoints=self.configuration['save_predictions_and_evaluate_on_labeled_samples']['metrics']['checkpoints'],
                                    thresholds=self.configuration['save_predictions_and_evaluate_on_labeled_samples']['metrics']['thresholds'],
                                    save_curves=self.configuration['save_predictions_and_evaluate_on_labeled_samples']['metrics']['save_curves']
                                )
            print('---------- Calculation of metrics on labeled samples is finished ------------')
        else:
            print("!!!---------- We don't calculate metrics for labeled samples ----------!!!")


    def postprocessing(self, thresholds=None, MF_sizes=None):
        data = np.load(os.path.join(self.saving_folder_with_checkpoint, self.whole_predictions_filename), allow_pickle=True)

        print(f'---------- We start postprocessing with MF sizes {MF_sizes} and thresholds {thresholds} ----------')
        original_filename = self.evaluator.comparable_characteristics_csvname
        original_metrics_filename = self.evaluator.metrics_filename_base

        for mf in MF_sizes:
            for threshold in thresholds:
                print(f'Postprocessing step: median filter size - {mf}, and threshold - {threshold}')
                folder_name = f"mf_{mf}_t_{threshold}"
                folder = os.path.join(self.saving_folder_with_checkpoint, folder_name)
                if not os.path.exists(folder):
                    os.mkdir(folder)

                postprocessed_predictions = {}

                for patient in data:
                    predictions_postprocessed = self.median_filter(patient, threshold, mf, folder)
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
                    
                self.evaluator.metrics_filename_base += '_postprocessed_' + str(mf)
                self.evaluator.metrics_filename_base = folder + config.SYSTEM_PATHS_DELIMITER \
                                                       + self.evaluator.metrics_filename_base
                self.evaluator.additional_columns = {'median': mf}

                thresholds_for_metrics = [threshold]
                if self.cross_validation_type == 'algorithm_plain':
                    thresholds_for_metrics = self.configuration['postprocessing']['thresholds']

                self.evaluator.evaluate(
                    save_curves=False,
                    predictions_npy_filename=folder_name + config.SYSTEM_PATHS_DELIMITER + self.file_with_postprocessed_predictions_for_labeled_samples,
                    thresholds=thresholds_for_metrics,
                    checkpoints=self.configuration['save_predictions_and_evaluate_on_labeled_samples']['metrics']['checkpoints']
                )

                self.evaluator.metrics_filename_base = original_metrics_filename
        self.evaluator.comparable_characteristics_csvname = original_filename
        self.evaluator.additional_columns = {}

        print(f'---------- Postprocessing is finished ----------')

    def median_filter(self, patient, threshold, median_filter_size, folder):
        number_of_classes = len(config.LABELS_OF_CLASSES_TO_TRAIN)

        size = patient['size']
        predictions = np.array(patient['predictions'])
        if number_of_classes > 2:
            predictions = np.argmax(predictions, axis=1)
        predictions = np.reshape(predictions, size)

        if threshold != -1:
            predictions[predictions >= threshold] = 1
            predictions[predictions < threshold] = 0

        predictions_postprocessed = median_filter(predictions, size=median_filter_size)
                
        if number_of_classes > 2: # we duplicate postprocessed labels for consistency of evaluation 
            predictions_postprocessed = np.eye(number_of_classes)[predictions_postprocessed.flatten()]
            predictions_postprocessed = np.reshape(predictions_postprocessed, list(size) + [number_of_classes])
                    
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

        np.save(os.path.join(folder, self.file_with_postprocessed_predictions_for_labeled_samples), result)
        
# old function for cross_validation.py to test models for post-processing papers        
def postprocessing_test_all_models():
    config.CV_GET_CHECKPOINT_FROM_VALID = False

    for model, scaling_type, threshold, checkpoint in zip([  
        'CV_3d_inception',
        #'CV_3d_inception_exclude1_all',
        #'CV_3d_inception_svn_every_third',
        #'CV_3d_svn_every_third',
        #'CV_3d_sample_weights_every_third',
        #'CV_3d_every_third',
        #'CV_3d_inception_exclude1_every_third',
    ], [  'l2_norm',
        #'svn_T', 
        #'svn_T',
        #'svn_T', 
        #'svn_T',
        #'l2_norm', 
        #'svn_T'
    ],
            [  0.2111,
                #0.0189, 
                #0.0456,
                #0.0367, 
                #0.45,
                #0.1556, 
                #0.0456
            ],
            [  36,
                #16, 
                #18,
                #16, 
                #18,
                #16, 
                #16
            ]):

        config.RAW_NPZ_PATH = os.path.join('/work/users/mi186veva/data_3d', scaling_type)
        config.NORMALIZATION_TYPE = scaling_type

        if scaling_type == 'svn_T':
            thresholds_range = [0.00001, threshold, 20]
        else:
            thresholds_range = [threshold - (threshold / 2), threshold + (threshold / 2), 20]
            #thresholds_range = [threshold, 2 * threshold, 20]

        cross_validator = get_cross_validator(
            model, cross_validation_type='algorithm_with_threshold', configuration={
                "generate_whole_cubes": False,
                # by default if "whole" folder is empty than we generate whole cubes, otherwise we don't. But with generate_whole_cubes it's possible to forse generate
                "calculate_predictions_for_whole_cubes": False,
                # by default if there is no predictions_whole.npy in test/name/cp-0000 than we count predoctions for whole cubes, otherwise - we don't. But with calculate_predictions_for_whole_cubes it's possible to forse count
                "save_predictions_and_evaluate_on_labeled_samples": {
                    # for detailed documentation of params in this dictionary see documentation for evaluation/evaluation_base.py/EvaluationBase.save_predictions_and_metrics()
                    "save_predictions": False,
                    "metrics": {
                        'save_metrics': False,
                        'checkpoints_range': None,
                        'checkpoints_raw_list': [checkpoint],
                        'thresholds_range': None,
                        'thresholds_raw_list': None,
                        'save_curves': False
                    }
                },
                "check": {  # what thresholds and median filter sizes to check
                    "median_filters_raw_list": [5, 25, 51],#[31, 35, 41, 45, 51, 55, 61, 65], #[5, 11, 15, 21, 25, 31, 35, 41, 45, 51, 55, 61],
                    "median_filters_range": None,
                    "thresholds_raw_list": None,#[0.1056, 0.2111, 0.3166],
                    "thresholds_range": thresholds_range

                }
            })
        cross_validator.evaluator.checkpoint_basename += scaling_type + '_'

        cross_validator.saving_folder_with_checkpoint = os.path.join(cross_validator.saving_folder,
                                                                     f'cp-{scaling_type}_{checkpoint:04d}')

        execution_flags = cross_validator.get_execution_flags()
        execution_flags['cross_validation'] = False
        cross_validator.pipeline(execution_flags=execution_flags)

        # utils.send_tg_message(f'Mariia, Post-processing for {model} is successfully completed!')
