import config
from data_loaders.data_loader_colon import DataLoaderColon
from data_loaders.data_loader_mat import DataLoaderMat


def get_data_loader(**kwargs):
    if config.DATABASE == 'colon':
        return DataLoaderColon(**kwargs)
    if config.DATABASE == 'bea_first_colon':
        return DataLoaderMat(**kwargs)

    raise ValueError('Error! Database type specified wrong (either in config.py or in data_loader.py)')


if __name__ == '__main__':
    dataLoader = get_data_loader()
    dataLoader.files_read_and_save_to_npz('/work/users/mi186veva/data', '/work/users/mi186veva/data_preprocessed/raw')
