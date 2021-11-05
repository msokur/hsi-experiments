#import cross_calidation
import glob
import csv
import os
import numpy as np
from tqdm import tqdm

class Validator():

    def __init__(self):
        return 

    def count_ROC_values(self):
        return 0

    def draw_ROC(self):
        return 0
    
    def generate_report(self):
        return 0
    
    def find_best_checkpoint(self, root_path, 
                             checkpoints_regex='cp-*', 
                             thresholds_filename='metrics_threshold_relation_by_patient.csv', 
                             threshold_raw=1, 
                             sens_spec_raws=[4, 5],
                             prints=True):
        checkpoints = glob.glob(os.path.join(root_path, checkpoints_regex))
        sorted(checkpoints)
        thresholds = []
        means = []
        i_sens = sens_spec_raws[0]
        i_spec = sens_spec_raws[1]
        
        for checkpoint in tqdm(checkpoints):
            print(checkpoint)
            metrics_csv_path = os.path.join(checkpoint, thresholds_filename)
        
            with open(metrics_csv_path, newline='') as f:
                reader = csv.reader(f, delimiter=',')
                data = []
                for row in reader:
                    data.append(row)
                data = np.array(data)
                thr = data[:, threshold_raw].astype(np.float32)
                sens = data[:, i_sens].astype(np.float32)
                spec = data[:, i_spec].astype(np.float32)
                idx = np.argwhere(np.diff(np.sign(sens - spec))).flatten()
                if prints:
                    print(idx, thr[idx])
                    print(sens[idx], spec[idx])
                    print(np.mean([sens[idx], spec[idx]]))
                
                thresholds.append(thr[idx])
                means.append(np.mean([sens[idx], spec[idx]]))
        
        
        print('------------------------------------------------------')
        best_idx = np.nanargmax(means)
        best_checkpoint = checkpoints[best_idx]
        best_threshold = thresholds[best_idx]

        print(f'Best index: {best_idx}, best checkpoint: {best_checkpoint}, best_threshold: {best_threshold}')
        print('Means:', means)
        print('------------------------------------------------------')
        
        return best_checkpoint, best_threshold, thresholds, means

    
if __name__ == '__main__':
    validator = Validator()
    validator.find_best_checkpoint('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/test/CV_best_4pat/')