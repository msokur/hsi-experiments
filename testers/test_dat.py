import numpy as np
import inspect
from sklearn.metrics import confusion_matrix, f1_score

import config
from testers.test_base import Tester
import data_utils.archive.data_loader_old as data_loader_old


class TesterDat(Tester):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def count_metrics(self, gt, predictions, name):
        print('--------------------method count_metrics params----------------------')
        signature = inspect.signature(Tester.count_metrics)
        for param in signature.parameters.values():
            print(param)
        print('------------------------------------------------')

        if len(predictions.shape) > 1:
            predictions = np.reshape(predictions, predictions.shape[0])

        predictions = predictions.astype(np.int)

        conf_matrix = confusion_matrix(gt, predictions, labels=config.LABELS_OF_CLASSES_TO_TRAIN)

        tn = conf_matrix[0, 0]
        tp = conf_matrix[1, 1]
        fn = conf_matrix[1, 0]
        fp = conf_matrix[0, 1]

        sensitivity = tp / (tp + fn)  # recall
        specificity = tn / (tn + fp)

        f1 = f1_score(gt, predictions)  # 2 * precision * sensitivity / (precision + sensitivity) #DICE score

        print('name', name, ', sensitivity: ', sensitivity, ', specificity: ', specificity, ', F1-score(DICE): ', f1)

        return sensitivity, specificity, f1

    def test_one_image(self, path, path_image=None, save=False, show=True, test_all_spectra=False, saving_path='',
                       grayscale_result=False, spectrum_shift=0):

        print('--------------------method test_one_image params----------------------')
        signature = inspect.signature(self.test_one_image)
        print('path_dat: ', path)
        for param in signature.parameters.values():
            print(param)
        print('------------------------------------------------')

        if saving_path == '':
            saving_path = self.SAVING_PATH

        # scaler = restore_scaler(FOLDER_NAME)

        gt_image, spectrum_data, healthy_indexes, ill_indexes, not_certain_indexes = \
            data_loader_old.get_data_for_showing(path, "")
        indexes = healthy_indexes + ill_indexes + not_certain_indexes
        indexes = np.array(indexes)

        if test_all_spectra:
            indexes = np.where(gt_image[:, :, 0] < 2055)
            indexes = np.array(indexes).T

        spectrum = spectrum_data[indexes[:, 0], indexes[:, 1]]

        if spectrum_shift != 0:
            spectrum = spectrum[..., :spectrum_shift]
        # spectrum = self.scaler.transform(spectrum)

        gt = [0] * len(healthy_indexes) + [1] * len(ill_indexes)
        gt = np.array(gt)

        # predictions = self.model.predict(np.expand_dims(spectrum, axis=-1))
        predictions = self.model.predict(spectrum)
        print('dsd gt.shape, predictions.shape', gt.shape, predictions.shape)
        print('dsd gt.unique, predictions.unique', np.unique(gt), np.unique(np.rint(predictions)))

        name = path.split(config.SYSTEM_PATHS_DELIMITER)[-1].split('_S')[0]

        sensitivity = specificity = 0

        if not test_all_spectra:

            sensitivity, specificity, _ = self.count_metrics(gt, np.rint(predictions), name)

            self.all_predictions += list(np.rint(predictions))
            self.all_predictions_raw += list(predictions)
            self.all_gt += gt

        self.save_or_show_result_prediction_on_image(predictions,
                                                     path_image,
                                                     saving_path,
                                                     name,
                                                     grayscale_result=grayscale_result,
                                                     save=save,
                                                     show=show)

        return sensitivity, specificity
