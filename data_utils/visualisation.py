import pandas as pd
import os
import inspect
import sys
from tqdm import tqdm
from glob import glob

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from configuration.get_config import DATALOADER, TRAINER, PATHS, CV
from data_utils.prediction_to_image.prediction_to_image_npz import PredictionToImage_npz
from data_utils.prediction_to_image.prediction_to_image_png import PredictionToImage_png


def get_csv_path(log_path: str, folder: str):
    path = os.path.join(*log_path, folder)
    csv_file = glob(os.path.join(path, "*.csv"))

    if len(csv_file) > 1:
        raise ValueError(f'Too many .csv files in "{path}"!')

    return csv_file[0]


def get_prediction_to_image(mode: str):
    if mode == "npz":
        return PredictionToImage_npz(load_conf=DATALOADER, model_conf=TRAINER)
    elif mode == "png":
        return PredictionToImage_png(load_conf=DATALOADER, model_conf=TRAINER)
    else:
        raise ValueError(f'No Mode "{mode}" found for visualisation!')


def get_model_path(path: str) -> str:
    paths = sorted(glob(os.path.join(path, PATHS["CHECKPOINT_PATH"], "cp-*")))

    return paths[-1]


def get_save_path(main_path: str, name: str) -> str:
    path = os.path.join(main_path, "visualisation")
    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, name + ".jpg")


def get_name_from_npz(path: str) -> str:
    return os.path.split(path)[-1].replace(".npz", "")


def get_dat_path(raw_path: str, name: str) -> str:
    if DATALOADER["NAME_SPLIT"] is not None:
        path = glob(os.path.join(raw_path, f"{name + DATALOADER['NAME_SPLIT']}.dat"))
    else:
        path = glob(os.path.join(raw_path, f"{name}.dat"))
    if len(path) > 1:
        raise ValueError(f'Too many .dat files with the name "{name}" in folder "{raw_path}"!')

    return path[0]


if __name__ == "__main__":
    pred_to_img = get_prediction_to_image(mode="npz")
    csv_path = get_csv_path(log_path=PATHS["MODEL_NAME_PATHS"], folder=CV["NAME"])
    csv_data = pd.read_csv(csv_path, delimiter=",", header=None, names=["Date", "x", "y", "z", "npz", "model"])

    for idx, row in tqdm(csv_data.iterrows()):
        file_name = get_name_from_npz(row["npz"])
        model_path = get_model_path(path=row["model"])
        npz_path = row["npz"]
        save_path = get_save_path(main_path=os.path.split(row["model"])[0], name=file_name)
        dat_path = get_dat_path(raw_path=PATHS["RAW_SOURCE_PATH"], name=file_name)
        anno_mask = pred_to_img.get_annotation_mask(path=npz_path)
        pred_mask = pred_to_img.get_prediction_mask(spectrum_path=npz_path, model_path=model_path)
        diff_mask = pred_to_img.get_diff_mask(annotation_mask=anno_mask, prediction_mask=pred_mask)

        print(f'Save Image for "{file_name}" in "{save_path}"')
        pred_to_img.save_with_background(save_path=save_path, background_path=dat_path, annotation_mask=anno_mask,
                                         predict_mask=pred_mask, diff_mask=diff_mask)
