import os

import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from data_utils.data_storage import DataStorage
from data_utils.hypercube_data import HyperCube

from configuration.keys import (
    DataLoaderKeys as DLK,
    PathKeys as PK
)


class VisualizationBase:
    def __init__(self, config, data_storage: DataStorage):
        self.config = config
        self.data_storage = data_storage
        self.labels = {label: color[0][:3] + [255]
                       for label, color in self.config.CONFIG_DATALOADER[DLK.MASK_COLOR].items()
                       if label in self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]}
        self.label_number = {index: color for index, color in enumerate(self.labels.values())}

    @staticmethod
    def _check_folder(path: str):
        if not os.path.exists(path):
            os.mkdir(path)

    def _get_rgb_cube(self, name: str) -> np.ndarray:
        path = glob(os.path.join(self.config.CONFIG_PATHS[PK.RAW_SOURCE_PATH], name + "*.dat"))
        cube_read = HyperCube(address=path[-1])
        cube = cube_read.cube_matrix(first_nm=self.config.CONFIG_DATALOADER[DLK.FIRST_NM],
                                     last_nm=self.config.CONFIG_DATALOADER[DLK.LAST_NM])

        return cube_read.rgb_cube(cube=cube)

    @staticmethod
    def _get_mask(shape: tuple, indexes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        mask = np.full(shape=shape, fill_value=-1)
        for index, label in zip(indexes, labels):
            mask[index[0], index[1]] = label

        return mask

    def _plot_masks(self, rgb_cube: np.ndarray, true_mask: np.ndarray, pred_mask: np.ndarray, error_mask: np.ndarray,
                    name: str, save_folder: str):
        headers = ["Annotation", "Classification", "Error Map"]

        fig = plt.figure(figsize=(14, 7), dpi=200)
        gs = fig.add_gridspec(nrows=3,
                              ncols=6)
        fig.suptitle(f"Visualisation from Patient {name}")
        rgba_cube = self._cube_to_rgba(rgb_cube=rgb_cube)

        for idx, (header, mask) in enumerate(zip(headers, [true_mask, pred_mask, error_mask])):
            col = 0 + 2 * idx
            ax = fig.add_subplot(gs[0:2, col: col + 2])
            ax.imshow(X=self._set_mask_background(rgba_cube=rgba_cube, mask=mask),
                      aspect="equal")
            plt.axis("off")
            ax.set_title(label=header)
        col_labels = ("Color", "Class", "Label")
        cell_colors, cell_text = [], []
        for (label, color), label_num in zip(self.labels.items(), self.label_number.keys()):
            cell_colors.append([[c / 255 for c in color], "w", "w"])
            cell_text.append(["", f"{label_num}", self.config.CONFIG_DATALOADER[DLK.TISSUE_LABELS][label]])
        ax_table = fig.add_subplot(gs[-1, 0:2])
        ax_table.table(cellText=cell_text,
                       cellColours=cell_colors,
                       colLabels=col_labels,
                       cellLoc="center",
                       loc="center")
        plt.axis("off")
        plt.savefig(os.path.join(save_folder, f"Error_map_pat_{name}.png"))
        plt.close(fig=fig)

    def _set_mask_background(self, rgba_cube: np.ndarray, mask: np.ndarray):
        rgba_mask = self._mask_to_rgba(mask=mask)

        return self._add_mask_to_cube(rgba_mask=rgba_mask, rgba_cube=rgba_cube)

    @staticmethod
    def _cube_to_rgba(rgb_cube: np.ndarray) -> np.ndarray:
        alpha_channel = np.ones(shape=rgb_cube.shape[:2] + (1,))
        rgba_cube = np.concatenate((rgb_cube, alpha_channel), axis=-1)
        rgba_cube *= 255

        return rgba_cube

    def _mask_to_rgba(self, mask: np.ndarray) -> np.ndarray:
        rgba_mask = np.zeros(shape=mask.shape + (4,))
        for index, color in self.label_number.items():
            rgba_mask[mask == index] = color

        return rgba_mask

    @staticmethod
    def _add_mask_to_cube(rgba_cube: np.ndarray, rgba_mask: np.ndarray) -> np.ndarray:
        new_image = (rgba_mask * 1.0 + rgba_cube * 1.0)
        new_image[new_image > 255.0] = 255.0

        return new_image.astype(int)
