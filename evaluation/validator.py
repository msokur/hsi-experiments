#import cross_calidation
import glob
import csv
import os
import numpy as np
from tqdm import tqdm

import config

class Validator():
    
    def __init__(self, root_path,
                         checkpoints_regex='cp-0000'+config.SYSTEM_PATHS_DELIMITER,
                         thresholds_filename='compare_all_thresholds.csv',
                         prints=True):
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
            print("sens_column_index, spec_column_index, threshold_column_index", sens_column_index, spec_column_index, threshold_column_index)
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

    def get_thresholds_sens_spec(self, data, sensitivity_column, specificity_column, threshold_column):
        # get data
        thr = data[:, threshold_column].astype(np.float32)
        sens = data[:, sensitivity_column].astype(np.float32)
        spec = data[:, specificity_column].astype(np.float32)

        return thr, sens, spec

    def get_values_by_threshold(self, checkpoint_folder, threshold):
        data, sensitivity_column, specificity_column, threshold_column = self.get_data(checkpoint_folder)
        thr, sens, spec = self.get_thresholds_sens_spec(data, sensitivity_column, specificity_column, threshold_column)
        idx = thr == threshold
        return sens[idx], spec[idx], np.mean([sens[idx], spec[idx]]), idx

    def find_best_threshold_in_checkpoint(self, checkpoint_folder):
        data, sensitivity_column, specificity_column, threshold_column = self.get_data(checkpoint_folder)

        thr, sens, spec = self.get_thresholds_sens_spec(data, sensitivity_column, specificity_column, threshold_column)

        #sort data for case if threshold are not sorted
        sorted_arrays = list(zip(list(thr), list(sens), list(spec)))
        sorted_arrays = sorted(sorted_arrays) #by default it sorts by the first column
        sorted_arrays = np.array(sorted_arrays)
        thr = sorted_arrays[:, 0]
        sens = sorted_arrays[:, 1]
        spec = sorted_arrays[:, 2]

        #print data
        #print('Sensitivities', sens)
        #print('Specificities', spec)

        #print intermediate computations
        #print(np.sign(sens - spec))
        diff = np.diff(np.sign(sens - spec))
        #print(diff)
        #print(np.argmin(diff))

        if len(diff[diff == -2]) == 0:
            print(f'~~~~~~~~~~~~~~~~You need to add new thresholds starts from {checkpoint_folder}~~~~~~~~~~~~~~')

        #get and print results
        idx = np.argmin(diff).flatten() + 1

        if self.prints:
            print(f'Best index and threshold for {checkpoint_folder}: ', idx, thr[idx])
            print(f'Best sensitivity and specificity for {checkpoint_folder}: ', sens[idx], spec[idx])
            print(f'Mean value of best sensitivity and specificity for {checkpoint_folder}: ', np.mean([sens[idx], spec[idx]]))

        return thr[idx], sens[idx], spec[idx], np.mean([sens[idx], spec[idx]]), idx
    
    def find_best_checkpoint(self, ):

        checkpoints = glob.glob(os.path.join(self.root_path, self.checkpoints_regex))
        if self.prints:
            print(os.path.join(self.root_path, self.checkpoints_regex))
            print(checkpoints)
        sorted(checkpoints)

        thresholds = []
        means = []
        best_sens = []
        best_spec = []

        for checkpoint in tqdm(checkpoints):
            threshold, sens, spec, mean, idx = self.find_best_threshold_in_checkpoint(checkpoint)
            best_sens.append(sens)
            best_spec.append(spec)
            thresholds.append(threshold)
            means.append(mean)
        
        
        best_idx = np.nanargmax(means)
        best_checkpoint = checkpoints[best_idx]
        best_threshold = thresholds[best_idx]

        if self.prints:
            print(f'Best index: {best_idx}, best checkpoint: {best_checkpoint}, best_threshold: {best_threshold}')
            print('Means:', means)
            print('------------------------------------------------------')
        
        return best_checkpoint, best_threshold, thresholds, means, best_sens[best_idx], best_spec[best_idx]

    
if __name__ == '__main__':
    validator = Validator()
    validator.find_best_checkpoint('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/test/CV_3d_inception/')