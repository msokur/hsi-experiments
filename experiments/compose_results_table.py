import os
from tqdm import tqdm
import numpy as np

import pandas as pd
from parser import Parser
import configuration.get_config as config
from evaluation.optimal_parameters import OptimalThreshold
from evaluation.metrics_csvreader import MetricsCsvReaderComparisonFiles


def compose_results(folders):
    column_names = ['Name',
                    'Folder',
                    'Size',
                    'Scaling',
                    'Sample_weights',
                    "Smoothing_dimension",
                    'Smoothing_algorithm',
                    'Smoothing_value',
                    'Background',
                    'Background_blood',
                    'Background_light',
                    'Threshold',
                    'Sensitivity',
                    'Specificity',
                    'Mean_between_sensitivity_and_specificity']
    table = []

    for folder in folders:
        for combination in tqdm(os.listdir(folder)):
            folder_path = os.path.join(folder, combination)
            if os.path.isdir(folder_path):
                parameters = Parser.parse_combination_parameters(combination)
                optimal_threshold_finder = OptimalThreshold(config, folder, prints=False)
                #print(combination)
                folder_with_checkpoint = os.path.join(folder, combination, "Results_with_EarlyStopping")
                results = optimal_threshold_finder.find_optimal_threshold_in_checkpoint(folder_with_checkpoint)
                '''data = MetricsCsvReaderComparisonFiles().read_metrics(os.path.join(folder_with_checkpoint,
                                                                          'compare_all_thresholds.csv'),
                                                                      names=['Threshold', 'Sensitivity', 'Specificity',
                                                                             'F1-score', 'MCC', 'AUC'])'''
                thr, sens, spec, _, _ = results

                row = [combination, folder] + list(parameters) + [thr, sens, spec, np.mean([sens, spec])]
                row = {k: v for k, v in zip(column_names, row)}

                table.append(row)

    df = pd.DataFrame(table)
    df.to_csv('results.csv')

#compose_results(['D:\\mi186veva-results\\MainExperiment_3_smoothing_2d'])
