from configuration.configloader_dataloader import read_dataloader_config


def test_read_dataloader_config(dataloader_data_dir: str, dataloader_result):
    assert read_dataloader_config(file=dataloader_data_dir, section="DATALOADER") == dataloader_result
