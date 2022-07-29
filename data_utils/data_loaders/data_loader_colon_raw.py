import numpy as np

from data_loader_raw_base import DataLoaderRawBase

class DataLoaderColonRaw(DataLoaderRawBase):
    def set_mask_with_labels(self, mask):       
        result_mask = np.zeros(mask.shape[:2]) - 1
        
        result_mask[(mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)] = 0  #blue = healthy
        result_mask[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)] = 1   #yellow = ill
        result_mask[(mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)] = 2   #red = not certain
        
        return result_mask
        