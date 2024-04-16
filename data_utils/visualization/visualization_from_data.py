import numpy as np
import os

from glob import glob

from data_utils.visualization.visualization_base import VisualizationBase
from data_utils.data_storage import DataStorage

from configuration.keys import PathKeys as PK

from configuration.parameter import (
    VISUALIZATION_FOLDER,
    DICT_y,
    DICT_IDX,
    ORIGINAL_NAME,
)


class VisualizationFromData(VisualizationBase):
    def __init__(self, config, data_storage: DataStorage):
        super().__init__(config, data_storage)

    def create_and_save_error_maps(self, save_path: str, threshold, y_true: np.ndarray, y_pred: np.ndarray,
                                   patient_name: str):
        if len(y_pred.shape) > 1:
            y_pred = y_pred.reshape(-1)

        save_folder = os.path.join(save_path, f"{VISUALIZATION_FOLDER}_by_threshold_{threshold}")
        self._check_folder(path=save_folder)

        for y_true_, y_pred_, cube_idx, name in self._get_data(patient_name=patient_name, y_true=y_true, y_pred=y_pred):
            rgb_cube = self._get_rgb_cube(name=name)
            true_mask = self._get_mask(shape=rgb_cube.shape[:2],
                                       indexes=cube_idx,
                                       labels=y_true_)
            pred_mask = self._get_mask(shape=rgb_cube.shape[:2],
                                       indexes=cube_idx,
                                       labels=y_pred_)
            error_mask = np.where(true_mask == pred_mask, -1, pred_mask)

            self._plot_masks(rgb_cube=rgb_cube,
                             true_mask=true_mask,
                             pred_mask=pred_mask,
                             error_mask=error_mask,
                             name=name,
                             save_folder=save_folder)

    def _get_data(self, patient_name: str, y_true: np.ndarray, y_pred: np.ndarray):
        data_path = glob(os.path.join(self.config.CONFIG_PATHS[PK.RAW_NPZ_PATH],
                                      patient_name + self.data_storage.get_extension()))[0]
        data = self.data_storage.get_datas(data_path=data_path)

        y = data[DICT_y]
        cube_idx = data[DICT_IDX]
        label_mask = np.isin(y, [label for label in self.label_number.keys()])

        self._check_values(values=y_true, mask=label_mask, name=patient_name)
        self._check_values(values=y_pred, mask=label_mask, name=patient_name)

        if ORIGINAL_NAME in data:
            names = np.unique(data[ORIGINAL_NAME])
            for name in names:
                name_mask = data[ORIGINAL_NAME][label_mask] == name
                mask = np.logical_and(label_mask, data[ORIGINAL_NAME] == name)
                yield y_true[name_mask], y_pred[name_mask], cube_idx[mask], name
        else:
            yield y_true, y_pred, cube_idx[label_mask], patient_name

    @staticmethod
    def _check_values(values: np.ndarray, mask: np.ndarray, name: str):
        if values.shape[0] != np.count_nonzero(mask):
            raise ValueError(f"Values from prediction.npy not match withe the values from the preprocessed file for "
                             f"patient {name}!")
