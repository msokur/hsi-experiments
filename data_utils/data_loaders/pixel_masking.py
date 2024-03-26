import os
import numpy as np
import provider
from configuration.keys import DataLoaderKeys as DLK
from data_utils.background_detection import detect_background


class PixelMasking:
    def __init__(self, config, path, spectrum, shape):
        self.config = config
        self.path = path
        self.spectrum = spectrum
        self.shape = shape

    def get_mask(self):
        pass

    def process_boolean_masks(self, boolean_masks):
        mask = self.get_mask()
        boolean_masks = [i * mask for i in boolean_masks]
        return boolean_masks


class Background(PixelMasking):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_mask = self.get_mask()

    def get_mask(self):
        background_mask = np.ones(self.shape).astype(bool)
        if self.config.CONFIG_DATALOADER["BACKGROUND"]["WITH_BACKGROUND_EXTRACTION"]:
            blood_threshold = self.config.CONFIG_DATALOADER["BACKGROUND"]["BLOOD_THRESHOLD"]
            lights_reflections_threshold = self.config.CONFIG_DATALOADER["BACKGROUND"]["LIGHT_REFLECTION_THRESHOLD"]
            background_mask = detect_background(self.spectrum,
                                                blood_threshold=blood_threshold,
                                                lights_reflections_threshold=lights_reflections_threshold)
            background_mask = np.reshape(background_mask, self.shape)

        return background_mask


class ContaminationMask(PixelMasking):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_mask(self):
        mask = np.full(self.shape, True)
        contamination_pht = os.path.join(self.path, self.get_contamination_filename())
        if os.path.exists(contamination_pht):
            import pandas as pd
            c_in = pd.read_csv(contamination_pht, names=["x-start", "x-end", "y-start", "y-end"], header=0, dtype=int)
            for idx in range(c_in.shape[0]):
                mask[c_in["y-start"][idx]:c_in["y-end"][idx], c_in["x-start"][idx]:c_in["x-end"][idx]] = False
        return mask

    def get_contamination_filename(self):
        return self.config.CONFIG_DATALOADER[DLK.CONTAMINATION_FILENAME]


class BorderMasking(PixelMasking):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_boolean_masks(self, masks, conf=None):
        if conf is None:
            conf = self.config.CONFIG_DATALOADER[DLK.BORDER_CONFIG]

        if conf[DLK.BC_ENABLE]:
            pixel_detect = provider.get_pixel_detection(conf[DLK.BC_METHODE])
            border_masks = []
            for idx, mask in enumerate(masks):
                if idx not in conf[DLK.BC_NOT_USED_LABELS]:
                    if len(conf[DLK.BC_AXIS]) == 0:
                        border_mask = pixel_detect(in_arr=masks[idx],
                                                   d=conf[DLK.BC_DEPTH])
                    else:
                        border_mask = pixel_detect(in_arr=masks[idx],
                                                   d=conf[DLK.BC_DEPTH],
                                                   axis=conf[DLK.BC_AXIS])
                    border_masks.append(border_mask)
                else:
                    border_masks.append(masks[idx])

            border_masks = [masks[i] * border_masks[i] for i in range(len(masks))]
            return border_masks

        return masks
