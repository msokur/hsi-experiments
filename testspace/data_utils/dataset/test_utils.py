import pytest

from glob import glob
import os

from data_utils.dataset.utils import parse_names_to_int

from configuration.parameter import (
    TFR_FILE_EXTENSION,
)

PARSE_NAME_TO_INT_DATA = [("tfr", "tfr", TFR_FILE_EXTENSION)]


@pytest.mark.parametrize("meta_type,folder,file_extension", PARSE_NAME_TO_INT_DATA)
def test_parse_names_to_int(data_dir: str, meta_type: str, folder: str, file_extension: str):
    path = os.path.join(data_dir, folder + "_file", "1d", "shuffled", "*" + file_extension)
    names_int = parse_names_to_int(files=sorted(glob(path)), meta_type=meta_type)
    assert names_int == {"test_0": 0, "test_1": 1, "test_2": 2, "test_3": 3, "test_4": 4}


@pytest.mark.parametrize("meta_type,folder,file_extension", PARSE_NAME_TO_INT_DATA)
def test_parse_names_to_int_error(data_dir: str, meta_type: str, folder: str, file_extension: str):
    paths = [os.path.join(data_dir, folder + "_file", "meta_files_error", f"shuffle{idx}{file_extension}")
             for idx in [0, 1]]
    with pytest.raises(ValueError, match="Too many patient indexes in meta files for the name 'test_4'!"):
        parse_names_to_int(files=paths, meta_type="tfr")
