import numpy as np

from data_utils.data_archive import DataArchive
from configuration.keys import DataLoaderKeys as DLK


class ChoicePaths:
    def __init__(self, data_archive: DataArchive, config_dataloader: dict, y_dict_name: str):
        self.data_archive = data_archive
        self.CONFIG_DATALOADER = config_dataloader
        self.y_dict_name = y_dict_name

    @staticmethod
    def random_choice(paths, excepts, size=1) -> np.ndarray:
        return np.random.choice([r for r in paths if r not in excepts],
                                size=size,
                                replace=False)

    def class_choice(self, paths, paths_names, excepts, classes=None) -> np.ndarray:
        if classes is None:
            classes = np.array([])
        valid = self.random_choice(paths_names, excepts)

        path_idx = paths_names.index(valid[0])
        unique_classes = self.data_archive.get_data(data_path=paths[path_idx], data_name=self.y_dict_name)
        con_classes = np.concatenate((classes, unique_classes[...]))
        con_unique_classes = np.intersect1d(con_classes, self.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN])
        if len(con_unique_classes) >= len(self.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]):
            return valid
        elif len(con_unique_classes) - len(classes) >= 1:
            return np.concatenate((valid, self.class_choice(paths,
                                                            paths_names,
                                                            np.concatenate((excepts, valid)),
                                                            con_unique_classes)))
        else:
            return self.class_choice(paths,
                                     paths_names,
                                     np.concatenate((excepts, valid)),
                                     classes)
