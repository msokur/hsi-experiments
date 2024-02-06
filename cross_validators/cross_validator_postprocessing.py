"""
Cross-validation with post-processing, which provides serious improvement of performance, especially if model classify
pixel by pixel.
Paper about post-processing: https://www.mdpi.com/2072-6694/15/7/2157
Documentation on wiki: https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/Post-processing

You can start post-processing with postprocessing_for_one_model() function in cross_validation.py
After initialization of CrossValidatorPostProcessing execution goes to CrossValidatorPostProcessing.evaluation()
MF - Median Filter
"""

import os
from glob import glob
import numpy as np
import matplotlib

matplotlib.use('Agg')

from scipy.ndimage import median_filter
from tqdm import tqdm
import csv

from data_utils.preprocessor import Preprocessor
from cross_validators.cross_validator_base import CrossValidatorBase
from provider import get_whole_analog_of_data_loader, get_evaluation
from configuration.keys import PathKeys as PK, CrossValidationKeys as CVK, DataLoaderKeys as DLK


class CrossValidatorPostProcessing(CrossValidatorBase):

    def __init__(self, config, **kwargs):
        self.config = config

        # default option for cross_validation_type is 'algorithm_plain', other option - 'algorithm_with_threshold',
        # but as pointed in documentation we recommend to use 'algorithm_plain'
        cross_validation_type = self.config.CONFIG_CV['POST_PROCESSING_ALGORITHM']
        if cross_validation_type == 'algorithm_with_threshold' and \
                len(self.config.CONFIG_DATALOADER['LABELS_TO_TRAIN']) > 2:
            raise ValueError(
                'Error! AWT (algorithm_with_threshold) could only be used with binary classification. Please use '
                '"algorithm_plain" for multiclass classification')

        super().__init__(config)
        self.LABELED_NPZ_FOLDER = self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH]
        self.cross_validation_type = cross_validation_type

        self.whole_database = get_whole_analog_of_data_loader(self.config.CONFIG_DATALOADER[DLK.TYPE])
        self.original_database = self.config.CONFIG_DATALOADER[DLK.TYPE]

        self.WHOLE_CUBES_FOLDER = os.path.join(self.LABELED_NPZ_FOLDER, 'whole')
        if not os.path.exists(self.WHOLE_CUBES_FOLDER):
            os.mkdir(self.WHOLE_CUBES_FOLDER)

        print('WHOLE_CUBES_FOLDER', self.WHOLE_CUBES_FOLDER)
        self.WHOLE_CUBES = glob(os.path.join(self.WHOLE_CUBES_FOLDER, '*.npz'))
        print('WHOLE_CUBES', self.WHOLE_CUBES)
        self.LABELED_CUBES = glob(os.path.join(self.LABELED_NPZ_FOLDER, '*.npz'))
        print('LABELED_CUBES', self.LABELED_CUBES)

        self.set_configuration(**kwargs)

        print('Post-processing configuration:', self.configuration)

        search_folder = os.path.join(self.config.CONFIG_PATHS[PK.LOGS_FOLDER], self.config.CONFIG_CV[CVK.NAME])
        self.training_csv_path = CrossValidatorBase.get_csv(search_folder)

        self.saving_folder = os.path.join(self.config.CONFIG_PATHS[PK.RESULTS_FOLDER], self.config.CONFIG_CV[CVK.NAME])
        if not os.path.exists(self.saving_folder):
            os.mkdir(self.saving_folder)

        self.saving_folder_with_checkpoint = os.path.join(self.saving_folder, 'cp-0000')
        if not os.path.exists(self.saving_folder_with_checkpoint):
            os.mkdir(self.saving_folder_with_checkpoint)

        self.evaluator = get_evaluation(labels=self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN],
                                        config=self.config)

        self.whole_predictions_filename = 'predictions_whole.npy'
        self.predictions_filename = self.evaluator.predictions_npy_filename
        self.file_with_postprocessed_predictions = 'predictions_postprocessed.npy'
        self.file_with_postprocessed_predictions_for_labeled_samples = 'predictions_postprocessed_for_labeled_samples.npy'

    @staticmethod
    def blank_configuration():
        return {
            # To apply post-processing first we have to calculate predictions for whole cubes
            # For this we need to generate whole cubes
            # by default if self.WHOLE_CUBES_FOLDER is emptier than we generate whole cubes, otherwise we don't.
            # But with generate_whole_cubes it's possible to force generation even if cubes are already generated
            "generate_whole_cubes": False,
            # by default if there is no predictions_whole.npy in metrics/training_name/cp-0000
            # than we calculate predictions for whole cubes, otherwise we don't. But with
            # calculate_predictions_for_whole_cubes it's possible to force calculation
            "calculate_predictions_for_whole_cubes": False,
            "save_predictions_and_evaluate_on_labeled_samples": {
                # standard evaluation parameters
                # for detailed documentation of params in this section see documentation for
                # evaluation/evaluation_base.py/EvaluationBase.save_predictions_and_metrics()
                "save_predictions": False,
                "metrics": {
                    'save_metrics': False,
                    'checkpoints': None,
                    'thresholds': None,
                    'save_curves': False
                }
            },
            "postprocessing": {  # what thresholds and median filter (MF) sizes to check
                # if classification is multiclass than specify only MF_sizes (because thresholds are not used)
                "MF_sizes": None,  # it's better to use odd MF sizes
                "thresholds": None,
            }
        }

    def set_configuration(self, **kwargs):
        if not kwargs:
            self.configuration = CrossValidatorPostProcessing.blank_configuration()
        else:
            self.configuration = kwargs['configuration']

    def evaluation(self, **kwargs):  # entry point

        self.generate_whole_cubes()
        self.calculate_predictions_on_whole_cubes()
        self.calculate_predictions_and_metrics_on_labeled_samples()

        if self.cross_validation_type == 'algorithm_with_threshold':
            thresholds = self.configuration['postprocessing']['thresholds']
        if self.cross_validation_type == 'algorithm_plain':
            # we fill thresholds with -1 to skip thresholding in CrossValidatorPostProcessing.median_filter() function
            thresholds = [-1]

        self.postprocessing(thresholds=thresholds, MF_sizes=self.configuration['postprocessing']['MF_sizes'])

    def generate_whole_cubes(self):
        if len(self.WHOLE_CUBES) != len(self.LABELED_CUBES) or self.configuration['generate_whole_cubes']:
            print('---------- Cubes generation is started------------')
            execution_flags = Preprocessor.get_execution_flags_for_pipeline_with_all_true()
            execution_flags['load_data_with_dataloader'] = True
            execution_flags['add_sample_weights'] = False
            execution_flags['scale'] = True
            execution_flags['shuffle'] = False

            self.config.CONFIG_DATALOADER[DLK.TYPE] = self.whole_database

            preprocessor = Preprocessor(self.config)
            preprocessor.pipeline(preprocessed_path=self.WHOLE_CUBES_FOLDER, execution_flags=execution_flags)
            self.config.CONFIG_DATALOADER[DLK.TYPE] = self.original_database

            print('---------- Cubes generation is finished------------')
        else:
            print(
                "!!!---------- We are not generating whole cubes, "
                "because they have already existed and 'generate_whole_cubes' is set to False ----------!!!")

    def calculate_predictions_on_whole_cubes(self):
        predictions_npz_exists = os.path.exists(os.path.join(self.saving_folder_with_checkpoint,
                                                             self.whole_predictions_filename))
        if not predictions_npz_exists or self.configuration['calculate_predictions_for_whole_cubes']:
            print('---------- Calculation of predictions on whole cubes is started------------')
            self.config.CONFIG_CV[CVK.USE_ALL_LABELS] = True

            self.evaluator.calculate_and_save_predictions(
                training_csv_path=self.training_csv_path,
                data_folder=self.WHOLE_CUBES_FOLDER,
                predictions_npy_filename=self.whole_predictions_filename
            )
            self.config.CONFIG_CV[CVK.USE_ALL_LABELS] = False
            print('---------- Calculation of predictions on whole cubes is finished ------------')
        else:
            print("!!!---------- We don't calculate predictions for whole cubes ----------!!!")

    def calculate_predictions_and_metrics_on_labeled_samples(self):
        predictions_exist = os.path.exists(os.path.join(self.saving_folder_with_checkpoint, self.predictions_filename))

        local_config = self.configuration['save_predictions_and_evaluate_on_labeled_samples']

        if not predictions_exist or local_config['save_predictions']:
            print('---------- Calculation of predictions on labeled samples is started ------------')
            self.evaluator.calculate_and_save_predictions(training_csv_path=self.training_csv_path,
                                                          data_folder=self.LABELED_NPZ_FOLDER,
                                                          predictions_npy_filename=self.predictions_filename,
                                                          checkpoints=local_config['metrics']['checkpoints'])

            print('---------- Calculation of predictions on labeled samples is finished ------------')
        else:
            print("!!!---------- We don't calculate predictions on labeled samples ----------!!!")

        if local_config['metrics']['save_metrics']:
            print('---------- Calculation of metrics on labeled predictions is started ------------')
            self.evaluator.evaluate(predictions_npy_filename=self.predictions_filename,
                                    checkpoints=local_config['metrics']['checkpoints'],
                                    thresholds=local_config['metrics']['thresholds'],
                                    save_curves=local_config['metrics']['save_curves'])
            print('---------- Calculation of metrics on labeled samples is finished ------------')
        else:
            print("!!!---------- We don't calculate metrics for labeled samples ----------!!!")

    def postprocessing(self, thresholds=None, MF_sizes=None):
        data = np.load(os.path.join(self.saving_folder_with_checkpoint, self.whole_predictions_filename),
                       allow_pickle=True)

        print(f'---------- We start postprocessing with MF sizes {MF_sizes} and thresholds {thresholds} ----------')
        original_filename = self.evaluator.comparison_csvname
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
                    predictions_postprocessed = self.median_filter(patient, threshold, mf)
                    postprocessed_predictions.update({
                        patient['name']: {
                            'predictions': predictions_postprocessed,
                            'gt': patient['gt'],
                        }
                    })

                np.save(os.path.join(folder, self.file_with_postprocessed_predictions), postprocessed_predictions)
                self.save_labeled_samples_from_postprocessed_whole_cubes(folder)

                self.evaluator.comparison_csvname = "compare_all_thresholds_postprocessed_AWT.csv"
                if self.cross_validation_type == 'algorithm_plain':
                    self.evaluator.comparison_csvname = "compare_all_thresholds_postprocessed_AP.csv"

                self.evaluator.metrics_filename_base += '_postprocessed_' + str(mf)
                self.evaluator.metrics_filename_base = folder + self.config.CONFIG_PATHS[PK.SYS_DELIMITER] \
                                                       + self.evaluator.metrics_filename_base
                self.evaluator.additional_columns = {'median': mf}

                thresholds_for_metrics = [threshold]
                if self.cross_validation_type == 'algorithm_plain':
                    thresholds_for_metrics = self.configuration['postprocessing']['thresholds']

                self.evaluator.evaluate(
                    save_curves=False,
                    predictions_npy_filename=folder_name + self.config.CONFIG_PATHS[
                        PK.SYS_DELIMITER] + self.file_with_postprocessed_predictions_for_labeled_samples,
                    thresholds=thresholds_for_metrics,
                    checkpoints=self.configuration['save_predictions_and_evaluate_on_labeled_samples']['metrics'][
                        'checkpoints']
                )

                self.evaluator.metrics_filename_base = original_metrics_filename
        self.evaluator.comparison_csvname = original_filename
        self.evaluator.additional_columns = {}

        print(f'---------- Postprocessing is finished ----------')

    def median_filter(self, patient, threshold, median_filter_size):
        number_of_classes = len(self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN])

        size = patient['size']
        predictions = np.array(patient['predictions'])
        if number_of_classes > 2:
            predictions = np.argmax(predictions, axis=1)
        predictions = np.reshape(predictions, size)

        if threshold != -1:
            predictions[predictions >= threshold] = 1
            predictions[predictions < threshold] = 0

        predictions_postprocessed = median_filter(predictions, size=median_filter_size)

        if number_of_classes > 2:  # we duplicate postprocessed labels for consistency of evaluation
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
                                                         self.file_with_postprocessed_predictions),
                                            allow_pickle=True).item()
        data_loader = get_data_loader(config=self.config, typ=self.config.CONFIG_DATALOADER[DLK.TYPE])
        result = []

        with open(self.training_csv_path, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                name = data_loader.get_name_func(row[4], delimiter='/')
                data = np.load(os.path.join(self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH], name + '.npz'))
                indexes_in_datacube = data['indexes_in_datacube']
                predictions = postprocessed_predictions[name]['predictions']
                predictions = predictions[indexes_in_datacube[:, 0], indexes_in_datacube[:, 1]]

                gt = data['y']
                indx_ = np.zeros(gt.shape).astype(bool)
                for label in self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]:
                    indx_ = indx_ | (gt == label)
                gt = gt[indx_]
                predictions = predictions[indx_]

                result.append({'name': name,
                               'gt': gt,
                               'predictions': predictions})

        np.save(os.path.join(folder, self.file_with_postprocessed_predictions_for_labeled_samples), result)
