import pytest

from configuration.configloader_paths import set_prefix, get_database_paths, read_path_config


@pytest.fixture
def path_prefix() -> str:
    return "/work/users/xyz"


@pytest.fixture
def base_path_data(path_prefix: str) -> dict:
    return {"CHECKPOINT_PATH": "checkpoints",
            "MODEL_PATH": "model",
            "MODE": "WITH_GPU",
            "PREFIX": path_prefix,
            "MODEL_NAME_PATHS": ["/home/sc.uni-leipzig.de/xyz/hsi-experiments-BA/logs"]}


@pytest.fixture
def prefix_data() -> dict:
    return {"1.CONCAT_WITH_PREFIX": {"RAW_NPZ_PATH": ["data_3d", "gastric", "3x3"],
                                     "RAW_SOURCE_PATH": ["Gastric"],
                                     "TEST_NPZ_PATH": ["data_3d", "hno", "3x3"]},
            "2.CONCAT_WITH_RAW_NPZ_PATH": {"SHUFFLED_PATH": ["shuffled"],
                                           "BATCHED_PATH": ["batch_sized"]},
            "3.CONCAT_WITH_RAW_SOURCE_PATH": {"MASK_PATH": ["annotation"]}}


@pytest.fixture
def concat_prefix_result(sys_slash: str, path_prefix: str):
    return {"RAW_NPZ_PATH": f"{path_prefix}{sys_slash}data_3d{sys_slash}gastric{sys_slash}3x3",
            "RAW_SOURCE_PATH": f"{path_prefix}{sys_slash}Gastric",
            "TEST_NPZ_PATH": f"{path_prefix}{sys_slash}data_3d{sys_slash}hno{sys_slash}3x3"}


@pytest.fixture
def get_database_result(base_path_data: dict, sys_slash: str, path_prefix: str, concat_prefix_result) -> dict:
    result = base_path_data.copy()
    result.update(concat_prefix_result)
    result.update({"SHUFFLED_PATH": f"{concat_prefix_result['RAW_NPZ_PATH']}{sys_slash}shuffled",
                   "BATCHED_PATH": f"{concat_prefix_result['RAW_NPZ_PATH']}{sys_slash}batch_sized",
                   "MASK_PATH": f"{concat_prefix_result['RAW_SOURCE_PATH']}{sys_slash}annotation"})
    return result


@pytest.fixture
def read_path_result(sys_slash: str, get_database_result: dict) -> dict:
    result = get_database_result.copy()
    result.update({"SYSTEM_PATHS_DELIMITER": sys_slash})
    return result


def test_read_path_config(paths_data_dir: str, read_path_result: dict):
    assert read_path_config(file=paths_data_dir, system_mode="SYSTEM", database="DATABASE") == read_path_result


def test_get_database_paths(base_path_data: dict, prefix_data: dict, get_database_result: dict):
    assert get_database_paths(path_dict=base_path_data, to_prefix=prefix_data) == get_database_result


def test_get_database_paths_error(base_path_data: dict, prefix_data: dict):
    revers_dict = dict(reversed(list(prefix_data.items())))
    with pytest.raises(ValueError,
                       match="Check the order form your Database paths, prefix: 'RAW_SOURCE_PATH' not found!"):
        get_database_paths(path_dict=base_path_data, to_prefix=revers_dict)


def test_set_prefix(path_prefix: str, prefix_data: dict, concat_prefix_result):
    assert set_prefix(prefix=path_prefix, database=prefix_data["1.CONCAT_WITH_PREFIX"]) == concat_prefix_result
