import pandas as pd
import numpy as np


class MetricsCsvReader:
    def fill_result(self, result, name, stacked_metrics):
        result[name] = {}
        result[name]['metrics'] = stacked_metrics[:-3]
        result[name]['mean'] = stacked_metrics[-3]
        result[name]['std'] = stacked_metrics[-2]
        result[name]['median'] = stacked_metrics[-1]
    
    def stack_columns(self, data, suitable_columns):
        return [np.array(data[column].tolist()).astype(float) for column in suitable_columns]
        
    def read_metrics(self, csvfile, names=[]):
        data = pd.read_csv(csvfile)
        columns = data.columns
        
        result = {}
        for name in names:
            suitable_columns = [column for column in columns if name in column]
            stacked_columns = self.stack_columns(data, suitable_columns)

            if len(suitable_columns) > 1:
                stacked_metrics = np.array([list(row) for row in zip(*stacked_columns)])
            else:
                stacked_metrics = stacked_columns[0]
            
            self.fill_result(result, name, stacked_metrics)
            
        return result

class MetricsCsvReaderComparisonFiles(MetricsCsvReader):   #for compare_all_thresholds*.csv files
    def fill_result(self, result, name, stacked_metrics):
        result[name] = stacked_metrics
    
    def stack_columns(self, data, suitable_columns):
        def convertable_to_float(string):
            try:
                result = float(string)
                return True
            except ValueError:
                return False
            
        stacked_columns = []
        for column in suitable_columns:
            rows = list(data[column])
            for row_i, row in enumerate(rows):
                if not convertable_to_float(row):
                    row = row.replace('[','')
                    row = row.replace(']','')
                    row = row.split(' ')
                    lst = [float(elem) for elem in row if elem != ''] 
                    rows[row_i] = lst
                stacked_columns.append(rows)
        return stacked_columns


    
if __name__ == '__main__':
    reader = MetricsCsvReader()
    print(reader.read_metrics('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/metrics/Esophagus_MedFilter/cp-0038/metrics_by_threshold_None.csv', names=['Sensitivity', 'Specificity', 'Accuracy']))
    #reader = MetricsCsvReaderComparisonFiles()
    #reader.read_metrics('/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/metrics/Esophagus_MedFilter/cp-0000/compare_all_thresholds.csv', names=['Sensitivity', 'Specificity'])