import numpy as np

from data_loader_whole_base import DataLoaderWholeBase


class DataLoaderWholeMat(DataLoaderWholeBase):
    def __init__(self, class_instance, **kwargs):
        super().__init__(class_instance, **kwargs)

    def set_mask_with_labels(self, mask):
        return mask