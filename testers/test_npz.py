import inspect
import numpy as np

import config
from tester_base import Tester


class TesterNpz(Tester):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def test_one_image(self, path, path_image=None, save=False, show=True, test_all_spectra=False,
                       saving_path='', grayscale_result=False,
                       spectrum_shift=0):

        print('--------------------method test_one_image params----------------------')
        signature = inspect.signature(self.test_one_image)
        print('path_dat: ', path)
        for param in signature.parameters.values():
            print(param)
        print('------------------------------------------------')

        if saving_path == '':
            saving_path = self.SAVING_PATH

        # scaler = restore_scaler(FOLDER_NAME)

        data = np.load(path)
        spectrum = data['X']
        # gt_image = cv2.imread(path_image)

        if spectrum_shift != 0:
            spectrum = spectrum[..., :spectrum_shift]
        # spectrum = self.scaler.transform(spectrum)

        gt = data['y']  # test batch
        indx_ = np.zeros(gt.shape).astype(bool)

        for label in config.LABELS_OF_CLASSES_TO_TRAIN:
            indx_ = indx_ | (gt == label)  # ((gt == 0) | (gt == 1))
        if config.WITH_BACKGROUND_EXTRACTION:
            gt = gt[indx_ & data['bg_mask']]
            spectrum = spectrum[indx_ & data['bg_mask']]

        gt = gt[indx_]
        spectrum = spectrum[indx_]

        # predictions = self.model.predict(np.expand_dims(spectrum, axis=-1))
        predictions = self.model.predict(spectrum)
        print('dsd gt.shape, predictions.shape', gt.shape, predictions.shape)
        print('dsd gt.unique, predictions.unique', np.unique(gt), np.unique(np.rint(predictions)))

        metrics = []
        name = path.split(config.SYSTEM_PATHS_DELIMITER)[-1].split('_S')[0]

        if not test_all_spectra:
            metrics = self.count_metrics(gt, np.rint(predictions), name)

            self.all_predictions += list(np.rint(predictions))
            self.all_predictions_raw += list(predictions)
            self.all_gt += list(gt)


        self.save_or_show_result_prediction_on_image(predictions,
                                                     path_image,
                                                     saving_path,
                                                     name,
                                                     grayscale_result=grayscale_result,
                                                     save=save,
                                                     show=show)

        return metrics
