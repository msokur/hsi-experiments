from data_utils._3d_patchifier import Patchifier
from data_utils.data_loaders.data_loader import DataLoader
import configuration.get_config as config
import provider
from configuration.parameter import (
    STORAGE_TYPE, DICT_X, DICT_y, DICT_IDX, ORIGINAL_NAME, BACKGROUND_MASK
)
import numpy as np
from configuration.keys import DataLoaderKeys as DLK

SPECTRUM_SIZE = config.CONFIG_DATALOADER[DLK.LAST_NM] - config.CONFIG_DATALOADER[DLK.FIRST_NM]
SIZE = config.CONFIG_DATALOADER[DLK.D3_SIZE]
TESTABLE_INDEX = [SIZE[0] // 2, SIZE[1] // 2]


def create_training_instances():
    instances = {
        DICT_X: np.random.rand(1, 92),
        DICT_y: [0],
        DICT_IDX: [TESTABLE_INDEX],
        ORIGINAL_NAME: ['Pat0'],
        BACKGROUND_MASK: None
    }
    return instances


def create_boolean_masks():
    masks = np.zeros([2] + SIZE)
    masks[0, *TESTABLE_INDEX] = 1
    return masks


def test_patchifier_algorithm():
    assert (_3D_training_instances[DICT_X][0] == spectrum).all()


def test_equality_of_items_of_instances():
    for key in [DICT_y, DICT_IDX, ORIGINAL_NAME]:
        assert _3D_training_instances[key][0] == training_instances[key][0]


def test_length_of_instances():
    for key in [DICT_X, DICT_y, DICT_IDX, ORIGINAL_NAME]:
        assert len(_3D_training_instances[key]) == len(training_instances[key])


def test_if_background_mask_is_inside():
    assert BACKGROUND_MASK in _3D_training_instances


data_storage = provider.get_data_storage(typ=STORAGE_TYPE)
data_loader = DataLoader(config, data_storage=data_storage)
patchifier = Patchifier(config)

training_instances = create_training_instances()
spectrum = np.random.rand(*SIZE, SPECTRUM_SIZE)
boolean_masks = create_boolean_masks()

_3D_training_instances = patchifier.get_3D_patches_onflow(training_instances=training_instances,
                                                          spectrum=spectrum,
                                                          boolean_masks=boolean_masks,
                                                          background_mask=None,
                                                          concatenate_function=data_loader.concatenate_train_instances)
