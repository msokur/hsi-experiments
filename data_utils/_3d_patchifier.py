from sklearn.feature_extraction import image
import numpy as np

from configuration.keys import DataLoaderKeys as DLK
from configuration.parameter import (
    DICT_X, DICT_IDX
)

# There are 2 approaches for patchifing: "Standard" and "Onflow".
# Standard first creates patches for the whole image using image.extract_patches_2d() function.
# Of course, it's very memory consuming, especially for patch sizes 7+,
# because (640, 480, 7, 7, 92) barely fits into memory.
# That's why exists "onflow": firstly 1D spectra are extracted.
# Then we iterate over 1D samples, look for its neighbours and construct 3D patch from neighbours
# This approach gives opportunity to check higher patch sizes, but it is slower than Standard for smaller patch sizes
# If you want to read more:
# https://git.iccas.de/MaktabiM/hsi-experiments/-/wikis/Research-conclusions/Comparing-standard-and-on-flow-patchifier


class Patchifier:
    def __init__(self, config):
        self.config = config

    def decide_3D_patchifier(self, size, dtype):
        use_standard_3D_patchifier = False
        use_onflow_3D_patchifier = False
        if self.config.CONFIG_DATALOADER[DLK.D3]:
            # 1Gb = 10**9 bytes
            # for explanation why we divide /1000 - check issue #75:
            # https://git.iccas.de/MaktabiM/hsi-experiments/-/issues/75

            spectrum_volume = np.prod(size) / 1000
            patch_volume = np.prod(self.config.CONFIG_DATALOADER[DLK.D3_SIZE]) / 1000
            bytes_per_element = np.dtype(dtype).itemsize / 1000

            needed_bytes = spectrum_volume * patch_volume * bytes_per_element
            needed_gb = needed_bytes

            if needed_gb > 5:
                use_onflow_3D_patchifier = True
                print(f'Onflow 3D patchifier will be used, because needed amount of Gb is {needed_gb}')
            else:
                use_standard_3D_patchifier = True
                print(f'Standard 3D  patchifier will be used. Needed amount of Gb is {needed_gb}')
        return use_standard_3D_patchifier, use_onflow_3D_patchifier

    def get_3D_patches_standard(self, spectrum: np.ndarray):
        size = self.config.CONFIG_DATALOADER[DLK.D3_SIZE]
        # Better not to use non even sizes
        pad = [int((s - 1) / 2) for s in size]
        pad_width = [[pad[idx], pad[idx]] if s % 2 == 1 else [pad[idx], pad[idx] + 1] for idx, s in enumerate(size)]
        pad_width.append([0, 0])
        spectrum_ = np.pad(array=spectrum, pad_width=np.array(pad_width))

        patches = image.extract_patches_2d(spectrum_, tuple(size))
        patches = np.reshape(patches, (spectrum.shape[0], spectrum.shape[1], size[0], size[1], patches.shape[-1]))

        return patches

    def get_3D_patches_onflow(self,
                              training_instances,
                              spectrum,
                              boolean_masks,
                              background_mask,
                              concatenate_function):
        train_indexes = training_instances[DICT_IDX]

        extended_boolean_masks = self.extend_masks(boolean_masks)
        extended_instances = concatenate_function(spectrum, extended_boolean_masks, background_mask)
        extended_spectrum = extended_instances['X']
        extended_indexes_in_datacube = extended_instances['indexes_in_datacube']

        training_instances[DICT_X] = self.patch_generator(extended_indexes_in_datacube,
                                                          train_indexes,
                                                          extended_spectrum,
                                                          spectrum)

        return training_instances

    def patch_generator(self, extended_indexes_in_datacube, train_indexes, extended_spectrum, spectrum):
        def get_condition(index):
            return (extended_indexes_in_datacube[:, index] >= (coordinates[index] - lookup)) & \
                   (extended_indexes_in_datacube[:, index] <= (coordinates[index] + lookup))

        sizes = self.config.CONFIG_DATALOADER[DLK.D3_SIZE]
        lookup = sizes[0] // 2

        result = np.zeros([train_indexes.shape[0], sizes[0], sizes[1], spectrum.shape[-1]])
        for index, coordinates in enumerate(train_indexes):
            x_condition = get_condition(0)
            y_condition = get_condition(1)

            pixels = extended_spectrum[x_condition & y_condition]
            indexes = extended_indexes_in_datacube[x_condition & y_condition]
            indexes[:, 0] -= coordinates[0] - lookup
            indexes[:, 1] -= coordinates[1] - lookup

            patch = np.zeros(sizes + [spectrum.shape[-1]])
            patch[tuple(indexes.T)] = pixels
            result[index] = patch

        return result

    def extend_masks(self, boolean_masks):
        import cv2
        output_boolean_masks = []
        for index, boolean_mask in enumerate(boolean_masks):
            margin = self.config.CONFIG_DATALOADER[DLK.D3_SIZE]
            kernel = np.ones((margin[0], margin[1]), np.uint8)

            contour_with_margin = cv2.dilate(boolean_mask.astype(np.int8) * 255, kernel)
            output_boolean_masks.append(np.array(contour_with_margin / 255).astype(bool))

        return output_boolean_masks
