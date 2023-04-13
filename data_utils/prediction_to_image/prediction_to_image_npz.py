import numpy as np

from data_utils.prediction_to_image.prediction_to_image_base import PredictionToImage_base


class PredictionToImage_npz(PredictionToImage_base):
    def get_spectrum(self, path: str):
        X, y, idx = self.load_npz(path=path)
        mask = np.isin(y, self.load_conf["LABELS_TO_TRAIN"])

        return X[mask], idx[mask]

    def get_annotation_mask(self, path: str) -> np.ndarray:
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


if __name__ == '__main__':
    import os
    from configuration.get_config import DATALOADER, TRAINER

    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    main_path = "D:\\ICCAS\\Daten\\predToImage\\paper_3x3_eso_median"
    model_path_ = os.path.join(main_path, "model_2019_04_30_15_34_56")

    image_path = os.path.join(main_path, "raw_data", "2019_04_30_15_34_56_SpecCube.png")
    # masks_png = test_png.annotation_mask(image_path)

    test_npz = PredictionToImage_npz(load_conf=DATALOADER, model_conf=TRAINER)
    npz_path = os.path.join(main_path, "raw_data", "2019_04_30_15_34_56_.npz")
    dat_path = r"E:\ICCAS\ESO\EsophagusCancer\2019_04_30_15_34_56_SpecCube.dat"
    anno_masks = test_npz.get_annotation_mask(npz_path)
    pred_mask = test_npz.get_prediction_mask(spectrum_path=npz_path, model_path=model_path_)
    diff_mask_ = test_npz.get_diff_mask(annotation_mask=anno_masks, prediction_mask=pred_mask)
    image_save = os.path.join(main_path, "mask.png")
    # test_npz.save_only_masks(image_save, anno_masks, pred_mask, diff_mask_)
    test_npz.save_with_background(image_save, dat_path, anno_masks, pred_mask, diff_mask_)
