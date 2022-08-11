import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from tqdm import tqdm
import csv

from data_utils.data_loaders.data_loader_base import DataLoader
from data_utils.preprocessor import Preprocessor
from cross_validators.cross_validator_base import CrossValidatorBase
import config
from validator import Validator

class CrossValidatorPostProcessing(CrossValidatorBase):
    def __init__(self, name, cross_validation_type='0_1', bea_db='bea_colon_whole', database='data_loader_whole_mat_colon', **kwargs):
        super().__init__(name)
        self.LABELED_NPZ_PATH = config.RAW_NPZ_PATH
        self.cross_validation_type = cross_validation_type
        self.bea_db = bea_db
        self.database = database

        self.WHOLE_CUBES_PATH = os.path.join(self.LABELED_NPZ_PATH, 'whole')
        if not os.path.exists(self.WHOLE_CUBES_PATH):
            os.mkdir(self.WHOLE_CUBES_PATH)

        print('WHOLE_CUBES_PATH', self.WHOLE_CUBES_PATH)
        self.WHOLE_CUBES = glob(os.path.join(self.WHOLE_CUBES_PATH, '*.npz'))
        print('WHOLE_CUBES', self.WHOLE_CUBES)
        self.LABELED_CUBES = glob(os.path.join(self.LABELED_NPZ_PATH, '*.npz'))
        print('LABELED_CUBES', self.LABELED_CUBES)

        if not kwargs:
            self.execution_flags = {
                "generate_whole_cubes": True,
                "get_predictions_for_whole_cubes": True,
                "count_predictions_for_labeled": True,
                "thr_ranges": [[0.1, 0.5, 20]]
            }
        else:
            self.execution_flags = kwargs['execution_flags']

       

        search_path = os.path.join(self.prefix, 'logs', name, '*.csv')
        print(search_path)
        self.csv_path = CrossValidatorBase.get_csv(search_path)

        self.saving_path = os.path.join(self.prefix, 'test', name)
        if not os.path.exists(self.saving_path):
            os.mkdir(self.saving_path)

        self.saving_path_with_checkpoint = os.path.join(self.saving_path, 'cp-0000')
        if not os.path.exists(self.saving_path_with_checkpoint):
            os.mkdir(self.saving_path_with_checkpoint)

        self.whole_predictions_filename = 'predictions_by_patient_whole.npy'
        self.whole_gt_filename = 'gt_by_patient_whole.npy'
        self.predictions_filename_ = 'predictions_by_patient.npy'
        self.gt_filename_ = 'gt_by_patient.npy'
        self.filtered_predictions_filename = 'predictions_by_patient_filtered.npy'
        self.filtered_gt_filename = 'gt_by_patient_filtered.npy'
        self.filtered_labeled_predictions_filename = 'predictions_by_patient_filtered_labeled.npy'
        self.filtered_labeled_gt_filename = 'gt_by_patient_filtered_labeled.npy'



    def evaluation(self, **kwargs):
        """ In kwargs
        execution_flags = {
        "generate_whole_cubes": True,
        "get_predictions_for_whole_cubes": True
        }
        """
        """
            Steps - 1 variant (median filter is applied to predictions 0-1):
            1. Generate whole cubes, if needed
            2. Make predictions for this whole cubes, if needed
            3. Make save_ROC_... for labeled and for thresholds, if needed
            4. Get the best threshold
            5. Due to this threshold create predictions map 0-1
            6. Apply median filter
            7. Get labeled with indexes_in_cube
            8. Count metrics again for the best threshold

            Steps - 2 varian (median filter is applied to raw predictions)
            1.-2. are the same
            3. Apply median filter on the raw predictions
            4. Get labeled with indexes_in_cube
            5. Make save_ROC...

        """

        self.generate_whole_cubes()
        self.count_predictions_on_whole_cubes()
        if self.cross_validation_type == '0_1':
            self.evaluation_0_1()
        if self.cross_validation_type == 'floats':
            self.evaluation_floats()

    def evaluation_0_1(self):
        self.count_predictions_on_labeled()

        best_checkpoint, best_threshold, thresholds, means = Validator().find_best_checkpoint(self.saving_path)
        print('Best threshold', best_threshold)

        self.median_filter(best_threshold)



        self.save_labeled_from_whole()

        self.metrics_for_threshold_filename_base += 'filtered'
        self.training_csv_file = self.csv_path
        self.predictions_filename = self.filtered_labeled_predictions_filename
        self.gt_filename = self.filtered_labeled_gt_filename
        config.USE_ALL_LABELS = False
        #config.bea_db = self.bea_db
        #config.DATABASE = config.DATABASES[self.database]

        self.count_metrics_on_diff_thresholds(self.saving_path_with_checkpoint, threshold_range_plain=best_threshold)
        print(self.csv_path)

        '''self.save_predictions_and_metrics_for_checkpoint(0,
                                                         self.saving_path,
                                                         self.csv_path,
                                                         thr_ranges=[[best_threshold, best_checkpoint, 1]],
                                                         execution_flags=[False],
                                                         predictions_filename=self.filtered_labeled_predictions_filename,
                                                         gt_filename=self.filtered_labeled_gt_filename
                                                         )'''





    def evaluation_floats(self):
        return

    def generate_whole_cubes(self):
        if len(self.WHOLE_CUBES) != len(self.LABELED_CUBES) or self.execution_flags['generate_whole_cubes']:
            execution_flags = Preprocessor.get_execution_flags_for_pipeline_with_all_true()
            execution_flags['load_data_with_dataloader'] = True
            execution_flags['add_sample_weights'] = False
            execution_flags['scale'] = True
            execution_flags['shuffle'] = False

            config.bea_db = self.bea_db
            config.DATABASE = config.DATABASES[self.database]

            preprocessor = Preprocessor()
            preprocessor.pipeline(config.RAW_SOURCE_PATH, self.WHOLE_CUBES_PATH,
                                  execution_flags=execution_flags)
        else:
            print("We don't generate whole cubes")

    def count_predictions_on_whole_cubes(self):
        pred_exists = os.path.exists(os.path.join(self.saving_path_with_checkpoint, self.whole_predictions_filename))
        gt_exists = os.path.exists(os.path.join(self.saving_path_with_checkpoint, self.whole_gt_filename))
        if not (pred_exists and gt_exists) or self.execution_flags['get_predictions_for_whole_cubes']:
            self.test_path = self.WHOLE_CUBES_PATH
            config.USE_ALL_LABELS = True

            self.save_predictions_and_metrics_for_checkpoint(0,
                                                             self.saving_path,
                                                             self.csv_path,
                                                             thr_ranges=[],
                                                             execution_flags=[True],
                                                             predictions_filename=self.whole_predictions_filename,
                                                             gt_filename=self.whole_gt_filename
                                                             )
        else:
            print("We don't count predictions for whole cubes")

    def count_predictions_on_labeled(self):
        print(self.saving_path, self.predictions_filename)
        pred_exists = os.path.exists(os.path.join(self.saving_path_with_checkpoint, self.predictions_filename_))
        gt_exists = os.path.exists(os.path.join(self.saving_path_with_checkpoint, self.gt_filename_))
        if not (pred_exists and gt_exists) or self.execution_flags['count_predictions_for_labeled']:
            self.test_path = self.LABELED_NPZ_PATH

            self.save_predictions_and_metrics_for_checkpoint(0,
                                                             self.saving_path,
                                                             self.csv_path,
                                                             thr_ranges=self.execution_flags['thr_ranges'],
                                                             execution_flags=[False],
                                                             predictions_filename=self.predictions_filename_,
                                                             gt_filename=self.gt_filename_
                                                             )
        else:
            print("We don't count predictions for labeled samples")
        return

    def median_filter(self, best_threshold):
        print('We start median filter')
        predictions_by_patient = np.load(
            os.path.join(self.saving_path_with_checkpoint, self.whole_predictions_filename), allow_pickle=True)
        gt_by_patient = np.load(os.path.join(self.saving_path_with_checkpoint, self.whole_gt_filename),
                                allow_pickle=True)

        filtered_predictions = []

        for patient in range(predictions_by_patient.shape[0]):
            print('pred', np.array(predictions_by_patient[patient]).shape)
            print('gt', np.array(gt_by_patient[patient]).shape)

            size = (640, 480)
            #size = (480, 640)
            pred = np.reshape(np.array(predictions_by_patient[patient])[:, 0], size)
            pred[pred >= best_threshold] = 1
            pred[pred < best_threshold] = 0
            gt = np.reshape(np.array(gt_by_patient[patient]), size)
            # print(np.unique(gt))
            gt_ = gt.copy()
            gt[gt_ == 1] = 2
            gt[gt_ == 0] = 1
            gt[gt_ == 2] = 0

            pred_filtered = median_filter(pred, size=55)
            #filtered_predictions.append(np.reshape(pred_filtered, size))
            filtered_predictions.append(pred_filtered)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=200)
            ax1.imshow(pred)
            ax2.imshow(pred_filtered)
            ax3.imshow(gt)
            plt.savefig(os.path.join(self.saving_path_with_checkpoint, str(patient) + '.png'))

        np.save(os.path.join(self.saving_path_with_checkpoint, self.filtered_predictions_filename),
                filtered_predictions)
        np.save(os.path.join(self.saving_path_with_checkpoint, self.filtered_gt_filename), gt_by_patient)



    def save_labeled_from_whole(self):
        filtered_predictions = np.load(os.path.join(self.saving_path_with_checkpoint,
                                                    self.filtered_predictions_filename))

        filtered_labeled_predictions = []
        gt_ = []

        with open(self.csv_path, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for p, row in tqdm(enumerate(report_reader)):
                name = DataLoader.get_name_easy(row[4], delimiter='/')
                data = np.load(os.path.join(config.RAW_NPZ_PATH, name+'.npz'))
                indexes_in_datacube = data['indexes_in_datacube']
                pred = filtered_predictions[p]
                print(pred.shape)
                pred = pred[indexes_in_datacube[:, 0], indexes_in_datacube[:, 1]]
                
                gt = data['y']
                print(gt.shape)
                print('unique 1', np.unique(gt))
                indx_ = np.zeros(gt.shape).astype(bool)
                for label in config.LABELS_OF_CLASSES_TO_TRAIN:
                    indx_ = indx_ | (gt == label)
                gt = gt[indx_]
                pred = pred[indx_]
                
                print('unique 2', np.unique(gt))
                print(gt.shape)

                filtered_labeled_predictions.append(pred)
                gt_.append(gt)

        np.save(os.path.join(self.saving_path_with_checkpoint, self.filtered_labeled_predictions_filename),
                filtered_labeled_predictions)
        np.save(os.path.join(self.saving_path_with_checkpoint, self.filtered_labeled_gt_filename), gt_)