import os
from tqdm import tqdm
import csv
import numpy as np

import config
from cross_validators.cross_validator_base import CrossValidatorBase
import utils


class CrossValidatorSpain(CrossValidatorBase):
    def __init__(self, name):
        if name == "":
            name = config.database_abbreviation
        super().__init__(name)

    def evaluation(self):
        name = self.name

        search_path = os.path.join(self.prefix, 'logs', name, '*.csv')
        csv_path = CrossValidatorBase.get_csv(search_path)
        # print(csv_path)

        saving_path = os.path.join('test', name)
        saving_path_whole = os.path.join('test', name + '_whole_image')
        if config.MODE == 'CLUSTER':
            saving_path = os.path.join(self.prefix, saving_path)
            saving_path_whole = os.path.join(self.prefix, saving_path_whole)

        nearest_int = self.get_nearest_int_delimiter(saving_path)

        best_checkpoint, best_checkpoints, model_paths = self.get_best_checkpoint_from_valid(csv_path,
                                                                                             nearest_int=nearest_int)
        print('best checkpoint', best_checkpoint)
        # print(best_checkpoints)
        # print(model_paths)

        if best_checkpoint > 0:

            # save predictions for whole image
            # self.test_path = config.TEST_NPZ_PATH
            '''cross_validator.save_ROC_thresholds_for_checkpoint(best_checkpoint,
                                                               saving_path_whole,
                                                               csv_path,
                                                               thr_ranges=[],
                                                               execution_flags=[False])'''

            # save annotated predictions
            self.test_path = config.RAW_NPZ_PATH
            self.save_predictions_and_metrics_for_checkpoint(0,
                                                             saving_path,
                                                             csv_path,
                                                             # thr_ranges=[],
                                                             thr_ranges=[
                                                                    # [0.001, 0.009, 10],
                                                                    [0.01, 0.09, 10],
                                                                    [0.1, 0.6, 10],
                                                                    # [0.75, 0.8, 10]
                                                                    # [0.45, 0.45, 1]
                                                                    # [0.15, 0.25, 10]
                                                                ],
                                                             execution_flags=[False])
        else:
            print('ATTENTION! Something during the comparing was wrong (probably history was empty)!')

    def get_best_checkpoint_from_valid(self, results_file, nearest_int=config.WRITE_CHECKPOINT_EVERY_Xth_STEP):
        model_paths = []
        best_checkpoints = []

        with open(results_file, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                model_path = row[5]
                if 'LOCAL' in config.MODE:
                    model_path = row[5].split('hsi-experiments')[-1][1:]

                model_paths.append(model_path)

                history, history_path = self.get_history(model_path)
                if not bool(history):
                    print(f'Attention!! {history_path} is empty!!!')
                    continue
                best_checkpoint = np.argmin(history[config.HISTORY_ARGMIN])
                best_checkpoints.append(best_checkpoint)

        if len(best_checkpoints) == 0:
            return -1, best_checkpoints, model_paths

        best_checkpoint = utils.round_to_the_nearest_even_int(np.median(best_checkpoints), nearest_int=nearest_int)

        return best_checkpoint, best_checkpoints, model_paths

    '''def cross_validation_spain(self, name=config.bea_db):

        # name = config.bea_db

        # csv_path = cross_validator.cross_validation(name)

        if name == 'ColonDatabase':
            csv_path = 'C:\\Users\\tkachenko\\Desktop\\HSI\\hsi-experiments\\logs\\CV_3d_bea_colon_sample_weights_1output\\CV_3d_bea_colon_sample_weights_1output_stats_16.12.2021-13_09_52.csv'
        else:
            # search_path = os.path.join(self.prefix, 'logs', name + '*', '*.csv')
            search_path = os.path.join(self.prefix, 'logs', name, '*.csv')
            csv_path = CrossValidatorBase.get_csv(search_path)
        # print(csv_path)

        test_path = os.path.join('test', name)
        test_path_whole = os.path.join('test', name + '_whole_image')
        if config.MODE == 'CLUSTER':
            test_path = os.path.join(self.prefix, test_path)
            test_path_whole = os.path.join(self.prefix, test_path_whole)

        nearest_int = self.get_nearest_int_delimiter(test_path)

        best_checkpoint, best_checkpoints, model_paths = self.get_best_checkpoint_from_valid(csv_path,
                                                                                             nearest_int=nearest_int)
        print('best checkpoint', best_checkpoint)
        # print(best_checkpoints)
        # print(model_paths)

        if best_checkpoint > 0:

            # save predictions for whole image
            self.test_path = config.TEST_NPZ_PATH
            cross_validator.save_ROC_thresholds_for_checkpoint(best_checkpoint,
                                                               test_path_whole,
                                                               csv_path,
                                                               thr_ranges=[],
                                                               execution_flags=[False])

            # save annotated predictions
            self.test_path = config.RAW_NPZ_PATH
            self.save_ROC_thresholds_for_checkpoint(0,
                                                    test_path,
                                                    csv_path,
                                                    thr_ranges=[
                                                        [0.01, 0.09, 10],
                                                        [0.1, 0.6, 10],
                                                    ],
                                                    execution_flags=[False])
        else:
            print('ATTENTION! Something during the comparing was wrong (probably history was empty)!')'''