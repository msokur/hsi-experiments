import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
#sys.path.insert(1, os.path.join(parentdir, 'utils')) 

import config
import utils

from  evaluation.evaluation_base import EvaluationBase
import datetime
import numpy as np
from glob import glob


class EvaluationBinary(EvaluationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_csv_fieldnames(self, metrics_dict, metrics_from_scores_dict):
        fieldnames = np.array(["Time"] + list(metrics_dict.keys()) + list(metrics_from_scores_dict.keys()),
                              dtype=object)
        dice_index = np.flatnonzero(fieldnames == "F1-score")
        fieldnames[dice_index] = "F1-score_healthy"
        fieldnames = np.insert(fieldnames, dice_index + 1, "F1-score_cancer")
        fieldnames = self.add_additional_column_fieldnames(fieldnames)

        return fieldnames

    def write_metrics_to_csv(self, writer, metrics, time_string=None):
        csv_row = {}
        dice_index = np.flatnonzero(np.array(list(metrics.keys())) == 'F1-score')[0]

        for metric_name, metric_value in list(metrics.items())[:dice_index] + list(metrics.items())[dice_index + 1:]:
            csv_row[metric_name] = metric_value
        csv_row["F1-score_healthy"] = metrics['F1-score'][0]
        csv_row["F1-score_cancer"] = metrics['F1-score'][1]
        if time_string is None:
            csv_row.update({"Time": datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")})
        else:
            csv_row.update({"Time": str(time_string)})
        self.write_additional_columns(csv_row)
        writer.writerow(csv_row)

    def calculate_predictions(self, predictions, threshold):
        return np.array(np.array(predictions) > threshold).astype(np.uint8)


if __name__ == '__main__':
    
    eval_binary = EvaluationBinary('Colon_MedianFilter')
    config.CV_GET_CHECKPOINT_FROM_VALID = False
    
    eval_binary.evaluate(#training_csv_path='/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/Colon_MedianFilter/Colon_MedianFilter_stats_02.02.2022-13_15_36.csv', 
                                               #npz_folder = os.path.join('/work/users/mi186veva', 'data_bea_db', 'Colon_MedianFilter', 'raw_3d_weighted'),
                                               checkpoints=[38],
                                                thresholds=np.round(np.linspace(0.001, 0.3, 10), 4))
    
    '''try:
    
        for folder, checkpoint, threshold, scaling_type in zip(['CV_3d_inception', 
                                       'CV_3d_inception_exclude1_all',
                                       'CV_3d_inception_svn_every_third',
                                       'CV_3d_svn_every_third',
                                       'CV_3d_sample_weights_every_third',
                                       'CV_3d_every_third',
                                       'CV_3d_bg_every_third',
                                       'CV_3d_inception_exclude1_every_third',
                                       'CV_3d'], [36, 16, 18, 16, 18, 16, 18, 16, 20],
                                     [0.2111,
                                        0.0189,
                                        0.0456,
                                        0.0367,
                                        0.45,
                                        0.1556,
                                        0.1556,
                                        0.0456,
                                        0.005], ['l2_norm',
                                                'svn_T',
                                                'svn_T',
                                                'svn_T',
                                                'svn_T',
                                                'l2_norm',
                                                'l2_norm',
                                                'svn_T',
                                                'l2_norm']):

            training_csv_paths = glob(os.path.join('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs', folder, '*.csv'))
            if len(training_csv_paths) == 0:
                print(f'ERROR!! No training_csv_paths in {folder}')
                continue
            if len(training_csv_paths) > 1:
                print(f'ERROR!! More then 1 training_csv_paths in {folder}')
                continue
            training_csv_path = training_csv_paths[0]

            #for scaling_type in ['l2_norm', 'svn', 'svn_T']:      

            eval_binary = EvaluationBinary(folder)

            eval_binary.checkpoint_basename += scaling_type + '_'

            eval_binary.save_predictions_and_metrics(training_csv_path=training_csv_path,
                                                     save_predictions=False,
                                                     npz_folder=os.path.join('/work/users/mi186veva/data_3d', scaling_type),
                                                     thresholds_raw_list=[threshold],
                                                     checkpoints_raw_list=[checkpoint],
                                                     save_curves=False)

            utils.send_tg_message(f'Mariia, evaluation for {folder} with {scaling_type} is successfully completed!')
                
        utils.send_tg_message(f'Mariia, evaluation_binary is successfully completed!')
               
    except Exception as e:

        utils.send_tg_message(f'Mariia, ERROR!!!, In evaluation error {e}')
        
        raise e'''
