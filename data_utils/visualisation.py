import pandas as pd
import os
import inspect
import sys
from tqdm import tqdm
from glob import glob

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from provider import get_data_archive, get_prediction_to_image
from configuration.get_config import CONFIG_DATALOADER, CONFIG_TRAINER, CONFIG_PATHS, CONFIG_CV
from configuration.keys import DataLoaderKeys as DLK, PathKeys as PK, CrossValidationKeys as CVK
from configuration.parameter import (
    ARCHIVE_TYPE, PRED_TO_IMG_TYP
)


def get_csv_path(log_path: str, folder: str):
    path = os.path.join(*log_path, folder)
    csv_file = glob(os.path.join(path, "*.csv"))

    if len(csv_file) > 1:
        raise ValueError(f'Too many .csv files in "{path}"!')

    return csv_file[0]


def get_model_path(path: str) -> str:
    paths = sorted(glob(os.path.join(path, CONFIG_PATHS[PK.CHECKPOINT_PATH], "cp-*")))

    return paths[-1]


def get_save_path(main_path: str, name: str) -> str:
    main_path_list = main_path.split(CONFIG_PATHS[PK.SYS_DELIMITER])
    # TODO create a variable in path config for test folder
    main_path_list[-2] = "test"
    path = CONFIG_PATHS[PK.SYS_DELIMITER] + os.path.join(*main_path_list, "visualisation")
    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, name + ".jpg")


def get_dat_path(raw_path: str, name: str) -> str:
    if CONFIG_DATALOADER[DLK.NAME_SPLIT] is not None:
        path = glob(os.path.join(raw_path, f"{name + CONFIG_DATALOADER[DLK.NAME_SPLIT]}.dat"))
    else:
        path = glob(os.path.join(raw_path, f"{name}.dat"))
    if len(path) > 1:
        raise ValueError(f'Too many .dat files with the name "{name}" in folder "{raw_path}"!')

    return path[0]


if __name__ == "__main__":
    pred_to_img = get_prediction_to_image(typ=PRED_TO_IMG_TYP, data_archive=get_data_archive(typ=ARCHIVE_TYPE),
                                          dataloader_conf=CONFIG_DATALOADER, model_conf=CONFIG_TRAINER)
    csv_path = get_csv_path(log_path=CONFIG_PATHS[PK.MODEL_NAME_PATHS], folder=CONFIG_CV[CVK.NAME])
    csv_data = pd.read_csv(csv_path, delimiter=",", header=None, names=["Date", "x", "y", "z", "archive", "model"])

    for idx, row in tqdm(csv_data.iterrows()):
        file_name = pred_to_img.get_name(path=row["archive"])
        model_path = get_model_path(path=row["model"])
        archive_path = row["archive"]
        save_path = get_save_path(main_path=os.path.split(row["model"])[0], name=file_name)
        dat_path = get_dat_path(raw_path=CONFIG_PATHS[PK.RAW_SOURCE_PATH], name=file_name)
        anno_mask = pred_to_img.get_annotation_mask(path=archive_path)
        pred_mask = pred_to_img.get_prediction_mask(spectrum_path=archive_path, model_path=model_path)
        diff_mask = pred_to_img.get_diff_mask(annotation_mask=anno_mask, prediction_mask=pred_mask)

        print(f'Save Image for "{file_name}" in "{save_path}"')
        pred_to_img.save_with_background(save_path=save_path, background_path=dat_path, annotation_mask=anno_mask,
                                         predict_mask=pred_mask, diff_mask=diff_mask)
