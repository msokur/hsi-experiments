import csv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from glob import glob
import os
import inspect

from configuration import get_config as conf
from configuration.keys import PathKeys as PK, TrainerKeys as TK, CrossValidationKeys as CVK, DataLoaderKeys as DLK
from models.model_randomness import set_tf_seed
from provider import get_data_loader, get_data_archive
from configuration.parameter import (
    ARCHIVE_TYPE
)

tf.random.set_seed(1)


class Predictor:
    """
    there are two variants:
    1. give LOGS_PATH and MODEL_PATH separately
    2. give MODEL_FOLDER
    """

    def __init__(self, CHECKPOINT, LOGS_PATH="", MODEL_NAME="", MODEL_FOLDER="",
                 custom_objects=conf.CONFIG_TRAINER[TK.CUSTOM_OBJECTS_LOAD]):

        if MODEL_NAME != '':
            self.MODEL_NAME = MODEL_NAME
        else:
            self.MODEL_NAME = MODEL_FOLDER.split(conf.CONFIG_PATHS[PK.SYS_DELIMITER])[-1]  # here can be problem

        if MODEL_FOLDER == '':
            MODEL_FOLDER = os.path.join(LOGS_PATH, self.MODEL_NAME)
            self.data_archive = get_data_archive(typ=ARCHIVE_TYPE)

        CHECKPOINTS_FOLDER_NAME = os.path.join(MODEL_FOLDER, "checkpoints")
        MODEL_PATH = os.path.join(CHECKPOINTS_FOLDER_NAME, CHECKPOINT)
        self.CHECKPOINT = CHECKPOINT

        set_tf_seed()
        self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

        print("--------------------PARAMS----------------------")
        print(", \n".join("%s: %s" % item for item in vars(self).items()))
        print("------------------------------------------------")

    def get_predictions_from_archive(self, path):
        data = self.data_archive.get_datas(data_path=path)
        spectrum = data["X"]
        gt = data["y"]
        size = None
        if "size" in data:
            size = data["size"]

        # get only needed samples
        indexes = np.full(shape=gt.shape, fill_value=False)
        for label in conf.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]:
            indexes = indexes | (gt == label)
        else:
            indexes = np.ones(gt.shape).astype(bool)
        if conf.CONFIG_DATALOADER[DLK.WITH_BACKGROUND_EXTRACTION] and "bg_mask" in data:
            gt = gt[indexes & data["bg_mask"]]
            spectrum = spectrum[indexes & data["bg_mask"]]

        gt = gt[indexes]
        spectrum = spectrum[indexes]

        predictions = self.model.predict(spectrum)

        return predictions, gt[...], size

    @staticmethod
    def edit_model_path_if_local(model_path):
        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in model_path:
            model_path = os.path.join(conf.CONFIG_PATHS[PK.MODEL_NAME_PATHS][0],
                                      model_path.split(conf.CONFIG_PATHS[PK.MODEL_NAME_PATHS][0])[-1][1:])
            if conf.CONFIG_PATHS[PK.SYS_DELIMITER] == "\\":
                model_path = model_path.replace("/", "\\")

            model_path = os.path.join(parent_dir, model_path)
        return model_path

    @staticmethod
    def get_best_checkpoint_from_csv(model_path):
        checkpoints_paths = sorted(glob(os.path.join(model_path,
                                                     conf.CONFIG_PATHS[PK.CHECKPOINT_PATH], "*")))
        best_checkpoint_path = checkpoints_paths[-1]
        return os.path.split(best_checkpoint_path)[-1]

    @staticmethod
    def get_checkpoint(checkpoint, model_path):
        if checkpoint is None:
            checkpoint = f"cp-{conf.CONFIG_TRAINER[TK.EPOCHS]:04d}"

        if conf.CONFIG_CV[CVK.GET_CHECKPOINT_FROM_VALID]:
            return Predictor.get_best_checkpoint_from_csv(model_path)
        else:
            return checkpoint

    def save_predictions(self, training_csv_path,
                         pat_archive_folder,
                         predictions_saving_folder,
                         predictions_npy_filename,
                         checkpoint=None):
        """
            param rows of training_csv_path:
            0 - date
            1 - index
            2 - sensitivity
            3 - specificity
            4 - .dat path
            5 - model path
        """

        results_dictionary = []
        data_loader = get_data_loader(typ=conf.CONFIG_DATALOADER[DLK.TYPE],
                                      data_archive=self.data_archive,
                                      config_dataloader=conf.CONFIG_DATALOADER,
                                      config_paths=conf.CONFIG_PATHS)
        with open(training_csv_path, newline='') as csvfile:
            report_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in tqdm(report_reader):
                print(', '.join(row))

                model_path = Predictor.edit_model_path_if_local(row[5])

                if checkpoint is not None:
                    checkpoint = Predictor.get_checkpoint(checkpoint, model_path)
                name = data_loader.get_name(path=row[4])
                print(f'We get checkpoint {checkpoint} for {model_path}')

                predictor = Predictor(checkpoint, MODEL_FOLDER=model_path)
                predictions, gt, size = predictor.get_predictions_from_archive(os.path.join(pat_archive_folder, name))

                results_dictionary.append({
                    'name': name,
                    'predictions': predictions,
                    'gt': gt,
                    'size': size
                })

        # saving of predictions
        np.save(os.path.join(predictions_saving_folder, predictions_npy_filename), results_dictionary)


if __name__ == "__main__":
    predictor_ = Predictor(f'cp-0020',
                           MODEL_FOLDER='/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/logs/CV_3d_inception'
                                        '/3d_0_1_2_3')
