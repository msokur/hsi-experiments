#import cross_calidation
import glob
import csv
import os
import numpy as np
from tqdm import tqdm

class Validator():
    
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
                
                #get data
                thr = data[:, threshold_raw].astype(np.float32)
                sens = data[:, i_sens].astype(np.float32)                
                spec = data[:, i_spec].astype(np.float32)                
                                
                #sort data for case if threshold are not sorted
                sorted_arrays = list(zip(list(thr), list(sens), list(spec)))
                sorted_arrays = sorted(sorted_arrays) #by default it sorts by the first column
                sorted_arrays = np.array(sorted_arrays)
                thr = sorted_arrays[:, 0]
                sens = sorted_arrays[:, 1]
                spec = sorted_arrays[:, 2]
                                
                #print data
                print('Sensitivities', sens)
                print('Specificities', spec)
                
                #print intermediate computations
                print(np.sign(sens - spec))
                diff = np.diff(np.sign(sens - spec))
                print(diff)
                print(np.argmin(diff))
                
                if len(diff[diff == -2]) == 0:
                    print(f'~~~~~~~~~~~~~~~~You need to add new thresholds starts from {checkpoint}~~~~~~~~~~~~~~')
                    
                #get and print results
                idx = np.argmin(diff).flatten() + 1
                if prints:
                    print('Best index and threshold: ', idx, thr[idx])
                    print('Best sensitivity and specificity: ', sens[idx], spec[idx])
                    print('Mean value of best sensitivity and specificity: ', np.mean([sens[idx], spec[idx]]))
                
                thresholds.append(thr[idx])
                means.append(np.mean([sens[idx], spec[idx]]))
        
        
        print('-----------------------')
        best_idx = np.nanargmax(means)
        best_checkpoint = checkpoints[best_idx]
        best_threshold = thresholds[best_idx]

        print(f'Best index: {best_idx}, best checkpoint: {best_checkpoint}, best_threshold: {best_threshold}')
        print('Means:', means)
        print('------------------------------------------------------')
        
        return best_checkpoint, best_threshold, thresholds, means

    
if __name__ == '__main__':
    validator = Validator()
    validator.find_best_checkpoint('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/test/CV_3d_svn_every_third/')