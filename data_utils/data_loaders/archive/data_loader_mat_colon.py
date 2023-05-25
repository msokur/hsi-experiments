import scipy.io

import config
from data_loader_mat import DataLoaderMat


class DataLoaderMatColon(DataLoaderMat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_number(self, elem):
        return elem.split("CP")[-1].split('.')[0]

    def get_labels(self):
        return [0, 1]

    def indexes_get_bool_from_mask(self, mask):
        ill_indexes = (mask == 1) 
        healthy_indexes = (mask == 2)

        return healthy_indexes, ill_indexes
