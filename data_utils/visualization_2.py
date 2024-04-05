from data_utils.data_storage import DataStorage
from data_utils.hypercube_data import HyperCube

from configuration.keys import DataLoaderKeys as DLK


class Visualization:
    def __init__(self, config, data_storage: DataStorage):
        self.config = config
        self.data_storage = data_storage
        self.labels = {label: color[0][:3] + [255]
                       for label, color in self.config.CONFIG_DATALOADER[DLK.LABELS].items()}
        self.label_number = {index: color for index, color in enumerate(self.labels.values())}

    def create_and_save_error_maps(self):
        pass
