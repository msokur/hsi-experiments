import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from glob import glob
from tensorflow import keras

from data_utils.visualization.visualization_base import VisualizationBase
from data_utils.data_storage import DataStorage
from models.model_base import ModelBase

from configuration.keys import (
    PathKeys as PK,
    CrossValidationKeys as CVK,
    TrainerKeys as TK
)

from configuration.parameter import (
    DICT_X,
    DICT_y,
    DICT_IDX,
    DICT_ORIGINAL_NAME,
    VISUALIZATION_FOLDER
)


class VisualizationFromCSV(VisualizationBase):
    def __init__(self, config, data_storage: DataStorage):
        super().__init__(config, data_storage)

    def create_and_save_error_maps(self, folder_name=VISUALIZATION_FOLDER):
        csv_path = self._get_csv_path(csv_folder=str(os.path.join(self.config.CONFIG_PATHS[PK.LOGS_FOLDER][0],
                                                                  self.config.CONFIG_CV[CVK.NAME])))
        csv_data = pd.read_csv(csv_path, delimiter=",", header=None,
                               names=["datetime", "index", "sensitivity", "specificity", "name", "model_name"])

        save_folder = str(os.path.join(self.config.CONFIG_PATHS[PK.RESULTS_FOLDER], self.config.CONFIG_CV[CVK.NAME],
                                       folder_name))
        self._check_folder(path=save_folder)

        for idx, row in tqdm(csv_data.iterrows()):
            model = self._get_model(path=row["model_name"])
            for X, y_true, cube_idx, name in self._get_data(storage_path=row["name"]):
                rgb_cube = self._get_rgb_cube(name=name)
                true_mask = self._get_mask(shape=rgb_cube.shape[:2],
                                           indexes=cube_idx,
                                           labels=y_true)
                pred_mask = self._get_predict_mask(model=model,
                                                   X=X,
                                                   shape=rgb_cube.shape[:2],
                                                   indexes=cube_idx)
                error_mask = np.where(true_mask == pred_mask, -1, pred_mask)

                self._plot_masks(rgb_cube=rgb_cube,
                                 true_mask=true_mask,
                                 pred_mask=pred_mask,
                                 error_mask=error_mask,
                                 name=name,
                                 save_folder=save_folder)

    def _get_csv_path(self, csv_folder: str) -> str:
        if CVK.CSV_FILENAME in self.config.CONFIG_CV:
            csv_file_path = os.path.join(csv_folder, self.config.CONFIG_CV[CVK.CSV_FILENAME])
        else:
            csv_file_path = os.path.join(csv_folder, "*.csv")

        csv_file = glob(csv_file_path)

        if len(csv_file) > 1:
            raise ValueError(f"Too many csv files in {csv_folder}. "
                             f"Set cvs filename in cross validation configurations!")
        elif len(csv_file) < 1:
            raise ValueError(f"No csv file found in {csv_folder}. Check your paths in the csv file!")

        return csv_file[0]

    def _get_data(self, storage_path: str):
        data = self.data_storage.get_datas(data_path=storage_path)
        data_name = self.data_storage.get_name(path=storage_path)

        X = data[DICT_X]
        y = data[DICT_y]
        cube_idx = data[DICT_IDX]
        label_mask = np.isin(y, [label for label in self.label_number.keys()])

        if DICT_ORIGINAL_NAME in data.keys():
            names = np.unique(data[DICT_ORIGINAL_NAME])
            for name in names:
                name_mask = data[DICT_ORIGINAL_NAME] == name
                mask = np.logical_and(name_mask, label_mask)
                yield X[mask], y[mask], cube_idx[mask], name
        else:
            for _ in range(1):
                yield X[label_mask], y[label_mask], cube_idx[label_mask], data_name

    def _get_model(self, path: str) -> keras.Model:
        paths = sorted(glob(os.path.join(path, self.config.CONFIG_PATHS[PK.CHECKPOINT_FOLDER], "cp-*")))

        return ModelBase.load_model(model_path=paths[-1],
                                    custom_objects=self.config.CONFIG_TRAINER[TK.CUSTOM_OBJECTS_LOAD])

    def _get_predict_mask(self, model: keras.Model, X: np.ndarray, shape: tuple, indexes: np.ndarray) -> np.ndarray:
        y_pred = model.predict(x=X)
        if len(y_pred.shape) > 1:
            if y_pred.shape[1] > 1:
                y_pred_1d = np.argmax(a=y_pred, axis=-1)
            else:
                y_pred_1d = np.array(y_pred > 0.5, dtype=np.uint8).reshape(-1)
        else:
            y_pred_1d = y_pred

        return self._get_mask(shape=shape, indexes=indexes, labels=y_pred_1d)


if __name__ == "__main__":
    from data_utils.data_storage import DataStorageNPZ
    import configuration.get_config as config_

    ds = DataStorageNPZ()

    visu = VisualizationFromCSV(config=config_, data_storage=ds)

    visu.create_and_save_error_maps()
