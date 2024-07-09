from data_utils._3d_patchifier import Patchifier
from data_utils.data_loaders.data_loader import DataLoader
from configuration.get_config import Config
import provider
from configuration.parameter import (
    STORAGE_TYPE, DICT_X, DICT_y, DICT_IDX, DICT_ORIGINAL_NAME, DICT_BACKGROUND_MASK
)
import numpy as np
from configuration.keys import DataLoaderKeys as DLK

SPECTRUM_SIZE = Config.CONFIG_DATALOADER[DLK.LAST_NM] - Config.CONFIG_DATALOADER[DLK.FIRST_NM]
SIZE = Config.CONFIG_DATALOADER[DLK.D3_SIZE]
TESTABLE_INDEX = [SIZE[0] // 2, SIZE[1] // 2]


def create_training_instances():
    instances = {
        DICT_X: np.random.rand(1, 92),
        DICT_y: np.array([0]),
        DICT_IDX: np.array([TESTABLE_INDEX]),
        DICT_ORIGINAL_NAME: np.array(['Pat0']),
        DICT_BACKGROUND_MASK: None
    }
    return instances


def create_boolean_masks():
    masks = np.zeros([len(Config.CONFIG_DATALOADER[DLK.LABELS])] + SIZE)
    masks[0, TESTABLE_INDEX[0], TESTABLE_INDEX[1]] = 1
    return masks


def test_patchifier_algorithm():
    assert (_3D_training_instances[DICT_X][0] == spectrum).all()


def test_equality_of_items_of_instances():
    for key in [DICT_y, DICT_IDX, DICT_ORIGINAL_NAME]:
        assert np.array_equal(_3D_training_instances[key][0], training_instances[key][0])


def test_length_of_instances():
    for key in [DICT_X, DICT_y, DICT_IDX, DICT_ORIGINAL_NAME]:
        assert len(_3D_training_instances[key]) == len(training_instances[key])


def test_if_background_mask_is_inside():
    assert DICT_BACKGROUND_MASK in _3D_training_instances


data_storage = provider.get_data_storage(typ=STORAGE_TYPE)
data_loader = DataLoader(Config, data_storage=data_storage)
patchifier = Patchifier(Config)

training_instances = create_training_instances()
spectrum = np.random.rand(*SIZE, SPECTRUM_SIZE)
boolean_masks = create_boolean_masks()

_3D_training_instances = patchifier.get_3D_patches_onflow(training_instances=training_instances,
                                                          spectrum=spectrum,
                                                          boolean_masks=boolean_masks,
                                                          background_mask=None,
                                                          concatenate_function=data_loader.concatenate_train_instances,
                                                          config=Config)
