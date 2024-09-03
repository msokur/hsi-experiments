import pytest

from configuration.configloader_paths import set_prefix, get_database_paths, read_path_config


@pytest.fixture
def prefix_data() -> dict:
    return {"0.CONCAT_WITH_DATABASE_ROOT_FOLDER": {"RAW_SOURCE_PATH": ["raw_database"]},
            "1.CONCAT_WITH_PREPROCESS_ROOT_FOLDER": {"ROOT_PATH": ["database", "3x3"]},
            "2.CONCAT_WITH_ROOT_PATH": {"RAW_NPZ_PATH": ["patient_data"],
                                        "SHUFFLED_PATH": ["shuffled"],
                                        "BATCHED_PATH": ["batch_sized"]},
            "3.CONCAT_WITH_RAW_SOURCE_PATH": {"MASK_PATH": ["annotation"]}}


@pytest.fixture
def get_database_result(path_result: dict) -> dict:
    return pop_from_dict(data=path_result, keys_to_pop=["SYSTEM_PATHS_DELIMITER"])


@pytest.fixture
def concat_prefix_result(get_database_result: dict) -> dict:
    return pop_from_dict(data=get_database_result, keys_to_pop=["SHUFFLED_PATH", "BATCHED_PATH", "RAW_NPZ_PATH",
                                                                "MASK_PATH", "LOGS_FOLDER", "PREPROCESS_ROOT_FOLDER",
                                                                "MODE", "RESULTS_FOLDER", "DATABASE_ROOT_FOLDER",
                                                                "CHECKPOINT_PATH", "RAW_SOURCE_PATH"])


@pytest.fixture
def base_path_data(get_database_result: dict) -> dict:
    return pop_from_dict(data=get_database_result, keys_to_pop=["RAW_NPZ_PATH", "RAW_SOURCE_PATH", "SHUFFLED_PATH",
                                                                "BATCHED_PATH", "MASK_PATH", "ROOT_PATH"])


def pop_from_dict(data: dict, keys_to_pop: list) -> dict:
    result = data.copy()
    list(map(result.pop, keys_to_pop))
    return result


def test_read_path_config(paths_data_dir: str, path_result: dict):
    assert read_path_config(file=paths_data_dir, system_mode="SYSTEM", database="DATABASE") == path_result


def test_get_database_paths(base_path_data: dict, prefix_data: dict, get_database_result: dict):
    assert get_database_paths(path_dict=base_path_data, to_prefix=prefix_data) == get_database_result


def test_get_database_paths_error(base_path_data: dict, prefix_data: dict):
    revers_dict = dict(reversed(list(prefix_data.items())))
    with pytest.raises(ValueError,
                       match="Check the order form your Database paths, prefix: 'RAW_SOURCE_PATH' not found!"):
        get_database_paths(path_dict=base_path_data, to_prefix=revers_dict)


def test_set_prefix(path_prepro_prefix: str, prefix_data: dict, concat_prefix_result):
    assert set_prefix(prefix=path_prepro_prefix, database=prefix_data["1.CONCAT_WITH_PREPROCESS_ROOT_FOLDER"]) == concat_prefix_result
