import pytest
import os

from data_utils.tfrecord import TFRDatasets


BATCH_SIZE = 5

GET_DATASETS_RANK = [("1d", False, [2, 1]), ("1d", True, [2, 1, 1]),
                     ("3d", False, [4, 1]), ("3d", True, [4, 1, 1])]


@pytest.mark.parametrize("shape,with_sw,ranks", GET_DATASETS_RANK)
def test_get_datasets_rank(tfr_data_dir: str, tfr_file_name: str, shape: str, with_sw: bool, ranks: list):
    tfr_datasets = TFRDatasets(batch_size=BATCH_SIZE, d3=True if shape == "3d" else False, with_sample_weights=with_sw)
    file = os.path.join(tfr_data_dir, shape, tfr_file_name)
    datasets = tfr_datasets.get_datasets(train_tfr_file=file, valid_tfr_file=file)
    for dataset in datasets:
        for element, rank in zip(dataset.element_spec, ranks):
            assert element.shape.rank == rank

