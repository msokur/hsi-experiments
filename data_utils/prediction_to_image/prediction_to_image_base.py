import abc
import os
import keras
import numpy as np
from matplotlib import colors as mcolor, pyplot as plt

from models.model_randomness import set_tf_seed
from data_utils.hypercube_data import cube

from configuration.keys import DataLoaderKeys as DLK, TrainerKeys as TK


class PredictionToImage_base:
    def __init__(self, dataloader_conf: dict, model_conf: dict, image_size=(480, 640)):
        self.CONFIG_DATALOADER = dataloader_conf
        self.CONFIG_MODEL = model_conf
        self.image_size = image_size

    def get_diff_mask(self, annotation_mask: np.ndarray, prediction_mask: np.ndarray) -> np.ndarray:
        diff_mask = np.full(self.image_size, -1)
        diff_mask[annotation_mask != prediction_mask] = prediction_mask[annotation_mask != prediction_mask]

        return diff_mask

    def get_prediction_mask(self, spectrum_path: str, model_path: str) -> np.ndarray:
        spectrum, indexes = self.get_spectrum(path=spectrum_path)
        model = self.load_model(model_path=model_path, custom_objects=self.CONFIG_MODEL[TK.CUSTOM_OBJECTS_LOAD])
        predict = model.predict(spectrum)
        predict_max = np.argmax(predict, axis=-1)
        predict_mask = np.full(self.image_size, -1)
        for idx, predict_class in zip(indexes, predict_max):
            predict_mask[idx[0]][idx[1]] = predict_class

        return self.check_classes(predict_mask=predict_mask)

    def check_classes(self, predict_mask: np.ndarray) -> np.ndarray:
        correct_mask = np.copy(predict_mask)
        if len(self.CONFIG_DATALOADER[DLK.MASK_COLOR]) != len(self.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]):
            key_range = range(sorted(self.CONFIG_DATALOADER[DLK.MASK_COLOR].keys())[-1])
            for label in key_range:
                if label not in self.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]:
                    correct_mask[correct_mask >= label] = correct_mask[correct_mask >= label] + 1

        return correct_mask

    def mask_to_rgba(self, mask: np.ndarray, mask_alpha=1.0) -> np.ndarray:
        img = np.zeros((self.image_size + (4,)))
        for label, color in self.CONFIG_DATALOADER[DLK.PLOT_COLORS].items():
            rgba = (mcolor.to_rgb(color) + (mask_alpha,))
            rgba = np.array(rgba) * 255
            img[mask == label] = rgba

        return np.array(img).astype(int)

    def save_with_background(self, save_path: str, background_path: str, annotation_mask: np.ndarray,
                             predict_mask: np.ndarray, diff_mask: np.ndarray, mask_alpha=0.75):
        masks_rgba = self.masks_to_rgba(masks=[annotation_mask, predict_mask, diff_mask])

        cube_ = cube(background_path, self.CONFIG_DATALOADER[DLK.WAVE_AREA], self.CONFIG_DATALOADER[DLK.FIRST_NM],
                     self.CONFIG_DATALOADER[DLK.LAST_NM])
        cube_img = cube_.get_rgb_cube()
        cube_img_rgb = cube_img * 255
        o_shape = cube_img_rgb.shape

        mixed_img = []
        for mask in masks_rgba:
            mixed_img.append(self.add_weighted(mask=mask[0:o_shape[0], 0:o_shape[1], 0:3], alpha=mask_alpha,
                                               original=cube_img_rgb, beta=1.0))

        self.save_image(save_path=save_path, masks=mixed_img)

    def save_only_masks(self, save_path: str, annotation_mask: np.ndarray, predict_mask: np.ndarray,
                        diff_mask: np.ndarray, mask_alpha=1.0):
        masks_rgba = self.masks_to_rgba(masks=[annotation_mask, predict_mask, diff_mask], mask_alpha=mask_alpha)

        self.save_image(save_path=save_path, masks=masks_rgba)

    def save_image(self, save_path: str, masks: list):
        fig = plt.figure(figsize=(14, 7), dpi=200)
        fig.suptitle(f"Visualisation from {os.path.split(save_path)[-1].split('.')[0]}", fontsize=16)
        gs = fig.add_gridspec(3, 6)
        labels = ["Annotation", "Classification", "Difference"]

        for idx, mask in enumerate(masks):
            ax = fig.add_subplot(gs[0:2, idx + 1 * idx: idx + 2 + 1 * idx])
            ax.imshow(mask, aspect="auto")
            plt.axis("off")
            ax.set_title(labels[idx])

        col_labels = ("Color", "Class", "Label")
        cell_colors, cell_text = [], []
        for label_num in sorted(self.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]):
            cell_colors.append([self.CONFIG_DATALOADER[DLK.PLOT_COLORS][label_num], "w", "w"])
            cell_text.append(["", f"{label_num}", self.CONFIG_DATALOADER[DLK.TISSUE_LABELS][label_num]])

        ax = fig.add_subplot(gs[-1, 0:2])
        ax.table(cellText=cell_text, cellColours=cell_colors, colLabels=col_labels, cellLoc="center",
                 loc="center")
        plt.axis("off")
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(save_path)
        plt.close(fig)

    def masks_to_rgba(self, masks: list, mask_alpha=1.0) -> list:
        masks_rgba = []
        for mask in masks:
            masks_rgba.append(self.mask_to_rgba(mask=mask, mask_alpha=mask_alpha))

        return masks_rgba

    @staticmethod
    def add_weighted(mask: np.ndarray, alpha: float, original: np.ndarray, beta: float) -> np.ndarray:
        new_img = (mask * alpha + original * beta)
        new_img[new_img > 255.0] = 255.0

        return new_img.astype(int)

    @abc.abstractmethod
    def get_spectrum(self, path: str):
        pass

    @abc.abstractmethod
    def get_annotation_mask(self, path: str):
        pass

    # TODO perhaps refactoring -> same code in evaluation.predictor.py
    @staticmethod
    def load_model(model_path: str, custom_objects=None) -> keras.Model:
        """
        Loads a keras model, return the model and set seed if necessary.

        :param model_path: The path to the model.
        :param custom_objects: A dict with custom metrics.

        Example
        -------
        custom_objects = {'F1_score': custom_metrics.F1_score}

        """
        set_tf_seed()
        model = keras.models.load_model(filepath=model_path, custom_objects=custom_objects)

        return model
