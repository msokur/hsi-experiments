import abc

import keras
import numpy as np
import cv2
from matplotlib import colors as mcolor, pyplot as plt
from matplotlib.gridspec import GridSpec

from models.model_randomness import set_tf_seed


class PredictionToImage_base:
    def __init__(self, load_conf: dict, model_conf: dict, image_size=(480, 640)):
        self.load_conf = load_conf
        self.model_conf = model_conf
        self.image_size = image_size

    def get_diff_mask(self, annotation_mask: np.ndarray, prediction_mask: np.ndarray) -> np.ndarray:
        diff_mask = np.full(self.image_size, -1)
        diff_mask[annotation_mask != prediction_mask] = prediction_mask[annotation_mask != prediction_mask]

        return diff_mask

    def get_prediction_mask(self, spectrum_path: str, model_path: str) -> np.ndarray:
        spectrum, indexes = self.get_spectrum(path=spectrum_path)
        model = self.load_model(model_path=model_path, custom_objects=self.model_conf["CUSTOM_OBJECTS_LOAD"])
        predict = model.predict(spectrum)
        predict_max = np.argmax(predict, axis=-1)
        predict_mask = np.full(self.image_size, -1)
        for idx, predict_class in zip(indexes, predict_max):
            predict_mask[idx[0]][idx[1]] = predict_class

        return self.check_classes(predict_mask=predict_mask)

    def check_classes(self, predict_mask: np.ndarray) -> np.ndarray:
        correct_mask = np.copy(predict_mask)
        if len(self.load_conf["MASK_COLOR"]) != len(self.load_conf["LABELS_TO_TRAIN"]):
            key_range = range(sorted(self.load_conf["MASK_COLOR"].keys())[-1])
            for label in key_range:
                if label not in self.load_conf["LABELS_TO_TRAIN"]:
                    correct_mask[correct_mask >= label] = correct_mask[correct_mask >= label] + 1

        return correct_mask

    def mask_to_rgba(self, mask: np.ndarray, mask_alpha=1.0) -> np.ndarray:
        img = np.zeros((self.image_size + (4,)))
        for label, color in self.load_conf["PLOT_COLORS"].items():
            rgba = (mcolor.to_rgb(color) + (mask_alpha,))
            rgba = np.array(rgba) * 255
            img[mask == label] = rgba

        return np.array(img).astype(int)

    def save_image_whole(self, path: str, annotation_mask: np.ndarray, predict_mask: np.ndarray, diff_mask: np.ndarray,
                         mask_alpha=1.0):
        fig = plt.figure(figsize=(36, 10), dpi=200)
        fig.patch.set_facecolor("ivory")
        rows, columns = 1, 4
        plt_offset = 1
        masks = [annotation_mask, predict_mask, diff_mask]
        labels = ["Annotation", "Classification", "Difference"]

        for idx in range(len(masks)):
            mask = self.mask_to_rgba(mask=masks[idx], mask_alpha=mask_alpha)
            fig.add_subplot(rows, columns, idx + plt_offset)
            plt.imshow(mask, aspect="auto")
            plt.axis("off")
            plt.title(labels[idx])

        col_labels = ("Color", "Class", "Label")
        cell_colors, cell_text = [], []
        for label_num in sorted(self.load_conf["LABELS_TO_TRAIN"]):
            cell_colors.append([self.load_conf["PLOT_COLORS"][label_num], "w", "w"])
            cell_text.append(["", f"{label_num}", self.load_conf["TISSUE_LABELS"][label_num]])

        fig.add_subplot(rows, columns, 4)
        the_table = plt.table(cellText=cell_text, cellColours=cell_colors, colLabels=col_labels, cellLoc="center",
                              loc="center")
        plt.axis("off")
        plt.subplots_adjust(wspace=0.0)
        plt.savefig(path)

    @abc.abstractmethod
    def get_spectrum(self, path: str):
        pass

    @abc.abstractmethod
    def annotation_mask(self, path: str):
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


class PredictionToImage_npz(PredictionToImage_base):
    def get_spectrum(self, path: str):
        X, _, idx = self.load_npz(path=path)

        return X, idx

    def annotation_mask(self, path: str) -> np.ndarray:
        _, y, idx = self.load_npz(path=path)
        mask = self.whole_mask(class_list=y, indexes=idx)

        return mask

    def whole_mask(self, class_list: list, indexes: list) -> np.ndarray:
        class_mask = np.full(self.image_size, -1)
        for idx, class_label in zip(indexes, class_list):
            if class_label not in self.load_conf["LABELS_TO_TRAIN"]:
                continue

            class_mask[idx[0]][idx[1]] = class_label
        return class_mask

    @staticmethod
    def load_npz(path: str):
        data = np.load(path)
        X = data["X"]
        y = data["y"]
        idx = data["indexes_in_datacube"]

        return X, y, idx


class PredictionToImage_png(PredictionToImage_base):
    def get_spectrum(self, path: str):
        pass

    def annotation_mask(self, path: str) -> np.ndarray:
        img = self.load_img(path=path)
        mask = self.whole_mask(img)

        return mask

    # TODO perhaps refactoring -> same code in data_utils.data_loaders.data_loaders_dyn.py
    @staticmethod
    def load_img(path: str) -> np.ndarray:
        """
                Loads an image and returns a numpy array with the RGBA Colorcode.

                :param path: The path from the image.

                :return: Returns an RGBA array from the Image.
                """
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # read Image with transparency
        # add alpha channel
        if mask.shape[-1] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)

        # [..., -2::-1] - BGR to RGB, [..., -1:] - only transparency, '-1' - concatenate along last axis
        mask = np.r_["-1", mask[..., -2::-1], mask[..., -1:]]

        return mask

    def whole_mask(self, mask: np.array) -> np.ndarray:
        """
        Create a mask with all class labels. -1 is not used indexes.

        :param mask: Annotation mask with RGB or RGBA color code.

        :return: Returns an array with all class labels.

        Example
        -------
        >>> color = {0: [[255, 0, 0]], 1: [[0, 0, 255]]}
        >>> a = [[[255, 0, 0], [255, 0, 0], [0, 255, 0]],
        ...      [[0, 255, 0], [0, 0, 255], [0, 255, 0]],
        ...      [[0, 0, 255], [0, 0, 255], [255, 0, 0]]]
        >>> b = whole_mask(a)
        array([ [0,     0,  -1],
        ...     [-1,    1,  -1],
        ...     [1,     1,   0]  ]

        """
        class_mask = np.full(mask.shape[0:2], -1)
        for key, value in self.load_conf["MASK_COLOR"].items():
            if key not in self.load_conf["LABELS_TO_TRAIN"]:
                continue

            for sub_value in value:
                if len(sub_value) == 3:
                    color_code = ((mask[..., 0] == sub_value[0]) & (mask[:, :, 1] == sub_value[1]) &
                                  (mask[:, :, 2] == sub_value[2]))
                elif len(sub_value) == 4:
                    color_code = ((mask[:, :, 0] == sub_value[0]) & (mask[:, :, 1] == sub_value[1]) &
                                  (mask[:, :, 2] == sub_value[2]) & (mask[:, :, 3] > sub_value[3]))
                else:
                    raise ValueError("Check your RGB/RGBA codes in configfile!")

                class_mask[color_code] = key

        return class_mask


if __name__ == '__main__':
    import os
    from configuration.get_config import DATALOADER, TRAINER

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    test_png = PredictionToImage_png(load_conf=DATALOADER, model_conf=TRAINER)

    main_path = "D:\\ICCAS\\Daten\\predToImage\\paper_3x3_eso_median"
    model_path_ = os.path.join(main_path, "model_2019_04_30_15_34_56")

    image_path = os.path.join(main_path, "raw_data", "2019_04_30_15_34_56_SpecCube.png")
    # masks_png = test_png.annotation_mask(image_path)

    test_npz = PredictionToImage_npz(load_conf=DATALOADER, model_conf=TRAINER)
    npz_path = os.path.join(main_path, "raw_data", "2019_04_30_15_34_56_.npz")
    anno_masks = test_npz.annotation_mask(npz_path)
    pred_mask = test_npz.get_prediction_mask(spectrum_path=npz_path, model_path=model_path_)
    diff_mask_ = test_npz.get_diff_mask(annotation_mask=anno_masks, prediction_mask=pred_mask)
    image_save = os.path.join(main_path, "mask.png")
    test_npz.save_image_whole(image_save, anno_masks, pred_mask, diff_mask_)

    x = 1
