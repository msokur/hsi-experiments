from configuration.configloader_cv import read_cv_config


def test_read_cv(cv_data_dir: str, cv_result):
    assert read_cv_config(file=cv_data_dir, base_section="BASE", section="CV") == cv_result
