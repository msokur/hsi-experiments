import glob
import csv
import os
import numpy as np
from tqdm import tqdm
from configuration.keys import CrossValidationKeys as CVK

import inspect
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


class OptimalThreshold:
    def __init__(self, config, root_path=None,
                 checkpoints_regex=None,
                 thresholds_filename='compare_all_thresholds.csv',
                 prints=True):
        self.config = config
        if checkpoints_regex is None:
            checkpoints_regex = 'cp-0000' + self.config.CONFIG_PATHS["SYSTEM_PATHS_DELIMITER"]

        if root_path is None:
            self.root_path = self.config.CONFIG_PATHS['RESULTS_FOLDER']
        else:
            self.root_path = root_path

        self.checkpoints_regex = checkpoints_regex
        self.thresholds_filename = thresholds_filename
        self.prints = prints

    def find_sens_spec_column(self, header):
        header = np.array([s.lower() for s in header])

        sens_column_index = np.where(header == "sensitivity_mean")[0][0]
        spec_column_index = np.where(header == "specificity_mean")[0][0]
        threshold_column_index = np.where(header == "threshold")[0][0]
        if self.prints:
            print("sens_column_index, spec_column_index, threshold_column_index", sens_column_index, spec_column_index,
                  threshold_column_index)
        return sens_column_index, spec_column_index, threshold_column_index

    def get_data(self, checkpoint_folder):
        if self.prints:
            print(checkpoint_folder)
        metrics_csv_path = os.path.join(checkpoint_folder, self.thresholds_filename)

        with open(metrics_csv_path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            data = []
            header = next(reader)
            if str.isdigit(header[0]):
                data.append(header)
                sensitivity_column = 4
                specificity_column = 5
                threshold_column = 2
            else:
                sensitivity_column, specificity_column, threshold_column = self.find_sens_spec_column(header)

            for row in reader:
                data.append(row)

            data = np.array(data)

        return data, sensitivity_column, specificity_column, threshold_column

    @staticmethod
    def get_thresholds_sens_spec(data, sensitivity_column, specificity_column, threshold_column):
        thr = data[:, threshold_column].astype(np.float32)
        sens = data[:, sensitivity_column].astype(np.float32)
        spec = data[:, specificity_column].astype(np.float32)

        return thr, sens, spec

    def get_values_by_threshold(self, checkpoint_folder, threshold):
        data, sensitivity_column, specificity_column, threshold_column = self.get_data(checkpoint_folder)
        thr, sens, spec = self.get_thresholds_sens_spec(data, sensitivity_column, specificity_column, threshold_column)
        idx = thr == threshold
        return sens[idx], spec[idx], np.mean([sens[idx], spec[idx]]), idx

    def check_if_thresholds_are_complete(self,
                                         thresholds,
                                         sensitivities,
                                         specificities,
                                         checkpoint_folder,
                                         optimal_index):
        difference_between_optimal_sens_and_spec = np.abs(sensitivities[optimal_index] -
                                                          specificities[optimal_index])
        additional_resolution_range_beginning = thresholds[optimal_index - 1]
        if optimal_index == 0:
            additional_resolution_range_beginning = thresholds[optimal_index]

        if optimal_index == len(thresholds) - 1:
            additional_resolution_range_end = thresholds[optimal_index]
        else:
            additional_resolution_range_end = thresholds[optimal_index + 1]

        completeness_options = {
            'add_thresholds_to_the_beginning': False,
            'add_thresholds_to_the_end': False,
            'existing_thresholds_range': [thresholds[0], thresholds[-1]],
            'difference_between_optimal_sens_and_spec': difference_between_optimal_sens_and_spec,
            'if_additional_resolution_for_optimal_threshold_is_needed': difference_between_optimal_sens_and_spec > 0.01,
            'additional_resolution_range': [additional_resolution_range_beginning, additional_resolution_range_end]
        }

        signs = np.sign(sensitivities - specificities)
        if_sensitivities_and_specificities_intersect = len(np.unique(signs)) > 1
        if not if_sensitivities_and_specificities_intersect:
            if signs[0] == 1:
                completeness_options['add_thresholds_to_the_end'] = True
                print(f'~~~~~~~~~~~~~~~~You need to add new thresholds to the end of the existing thresholds range '
                      f'({thresholds[0]} - {thresholds[-1]}) for {checkpoint_folder}~~~~~~~~~~~~~~')
            if signs[1] == -1:
                completeness_options['add_thresholds_to_the_beginning'] = True
                print(f'~~~~~~~~~~~~~~~~You need to add new thresholds to the beginning of the existing thresholds '
                      f'range ({thresholds[0]} - {thresholds[-1]}) for {checkpoint_folder}~~~~~~~~~~~~~~')

        return completeness_options

    def find_optimal_threshold_in_checkpoint(self, checkpoint_folder):
        data, sensitivity_column, specificity_column, threshold_column = self.get_data(checkpoint_folder)

        thresholds, sensitivities, specificities = self.get_thresholds_sens_spec(data,
                                                                                 sensitivity_column,
                                                                                 specificity_column,
                                                                                 threshold_column)

        # sort data for case if threshold are not sorted
        sorted_arrays = list(zip(list(thresholds), list(sensitivities), list(specificities)))
        sorted_arrays = sorted(sorted_arrays)  # by default, it sorts by the first column
        sorted_arrays = np.array(sorted_arrays)
        thresholds = sorted_arrays[:, 0]
        sensitivities = sorted_arrays[:, 1]
        specificities = sorted_arrays[:, 2]

        # print data
        # print('Sensitivities', sens)
        # print('Specificities', spec)

        differences = np.abs(sensitivities - specificities)
        optimal_index = np.argmin(differences)

        completeness_options = self.check_if_thresholds_are_complete(thresholds,
                                                                     sensitivities,
                                                                     specificities,
                                                                     checkpoint_folder,
                                                                     optimal_index)

        if self.prints:
            print(f'For checkpoint folder: {checkpoint_folder}:')
            print(f'- optimal index and threshold: ', optimal_index, thresholds[optimal_index])
            print(f'- optimal sensitivity and specificity: ', sensitivities[optimal_index],
                  specificities[optimal_index])
            print(f'- mean value of optimal sensitivity and specificity: ',
                  np.mean([sensitivities[optimal_index], specificities[optimal_index]]))

        return thresholds[optimal_index], \
               sensitivities[optimal_index], \
               specificities[optimal_index], \
               optimal_index, \
               completeness_options

    def add_additional_thresholds_if_needed(self, CV):
        def get_executions_flags():
            return {
                CVK.EF_CROSS_VALIDATION: False,
                CVK.EF_EVALUATION: True
            }

        def evaluate_additional_thresholds(begin, end):
            CV.pipeline(execution_flags=get_executions_flags(), save_predictions=False,
                        thresholds=np.linspace(begin, end, 10))

        def calculate_optimal_threshold():
            results = self.find_optimal_threshold_in_checkpoint(folder_with_checkpoint)
            threshold, sensitivity, specificity, _, completeness_options = results
            print(threshold, sensitivity, specificity, completeness_options)
            return completeness_options

        # self.config.CONFIG_PATHS['RESULTS_FOLDER'] = self.config.CONFIG_PATHS['RESULTS_FOLDER'].replace('Debug_thresholds', 'MainExperiment_3d_3_fixed_background') #TODO REMOVE!!!!!!!!!!!!

        folder_with_checkpoint = os.path.join(self.root_path,
                                              self.config.CONFIG_CV["NAME"],
                                              'Results_with_EarlyStopping')

        completeness_options = calculate_optimal_threshold()

        while completeness_options['add_thresholds_to_the_beginning']:
            begin = max(completeness_options['existing_thresholds_range'][0] - 0.1, 0)
            end = completeness_options['existing_thresholds_range'][0]
            print(
                f'-------------------Adding additional thresholds ({begin} - {end}) '
                f'to the beginning of the existing range-------------------')

            evaluate_additional_thresholds(begin=begin, end=end)

            completeness_options = calculate_optimal_threshold()

        while completeness_options['add_thresholds_to_the_end']:
            begin = completeness_options['existing_thresholds_range'][1]
            end = min(completeness_options['existing_thresholds_range'][1] + 0.1, 1)
            print(
                f'-------------------Adding additional thresholds ({begin} - {end}) '
                f'to the end of the existing range-------------------')

            evaluate_additional_thresholds(begin=begin, end=end)

            completeness_options = calculate_optimal_threshold()

        while completeness_options['if_additional_resolution_for_optimal_threshold_is_needed']:
            begin = completeness_options['additional_resolution_range'][0]
            end = completeness_options['additional_resolution_range'][1]
            print(
                f'-------------------Adding additional thresholds ({begin} - {end}) '
                f'to increase thresholds resolution-------------------')

            evaluate_additional_thresholds(begin=begin, end=end)

            completeness_options = calculate_optimal_threshold()


class OptimalCheckpoint(OptimalThreshold):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_optimal_checkpoint(self):

        checkpoints = glob.glob(os.path.join(self.root_path, self.checkpoints_regex))
        if self.prints:
            print(os.path.join(self.root_path, self.checkpoints_regex))
            print(checkpoints)
        sorted(checkpoints)

        thresholds = []
        means = []
        optimal_sens = []
        optimal_spec = []

        for checkpoint in tqdm(checkpoints):
            threshold, sens, spec, mean, idx = self.find_optimal_threshold_in_checkpoint(checkpoint)
            optimal_sens.append(sens)
            optimal_spec.append(spec)
            thresholds.append(threshold)
            means.append(mean)

        optimal_idx = np.nanargmax(means)
        optimal_checkpoint = checkpoints[optimal_idx]
        optimal_threshold = thresholds[optimal_idx]

        if self.prints:
            print(
                f'optimal index: {optimal_idx}, optimal checkpoint: {optimal_checkpoint}, '
                f'optimal_threshold: {optimal_threshold}')
            print('Means:', means)
            print('------------------------------------------------------')

        return optimal_checkpoint, optimal_threshold, thresholds, means, optimal_sens[optimal_idx], optimal_spec[
            optimal_idx]


if __name__ == '__main__':
    from configuration.get_config import CVConfig

    '''root_folder = "D:\\mi186veva-results\\MainExperiment_3d_3_fixed_background"
    optimal_threshold_finder = OptimalThreshold(config, root_folder, prints=False)
    folder_with_checkpoint = 'D:\\mi186veva-results\\MainExperiment_3d_3_fixed_background\\0_3D3_Ns_WT_Sm_S3_BF_B0' \
                            '.1_B0.25_\\Results_with_EarlyStopping'
    results = optimal_threshold_finder.find_optimal_threshold_in_checkpoint(folder_with_checkpoint)
    threshold, sensitivity, specificity, mean, completeness_options = results
    print(threshold, sensitivity, specificity, mean, completeness_options)'''

    '''from evaluation.metrics_csvreader import MetricsCsvReader
    reader = MetricsCsvReader()
    reader.read_metrics(
        '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/metrics/Esophagus_MedFilter/cp-0038'
        '/metrics_by_threshold_None.csv',
        names=['Sensitivity', 'Specificity'])'''
