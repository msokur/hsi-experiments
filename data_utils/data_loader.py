import config
from data_loaders.data_loader_easy import DataLoaderColon
from data_loaders.data_loader_mat import DataLoaderMat
from data_loaders.data_loader_mat_brain import DataLoaderMatBrain
from data_loaders.data_loader_mat_colon import DataLoaderMatColon


def get_data_loader(**kwargs):
    if config.DATABASE == 'colon':
        print('DataLoaderColon')
        return DataLoaderColon(**kwargs)
    if config.DATABASE == 'bea_eso':
        print('DataLoaderMat')
        return DataLoaderMat(**kwargs)
    if config.DATABASE == 'bea_brain':
        print('DataLoaderMatBrain')
        return DataLoaderMatBrain(**kwargs)
    if config.DATABASE == 'bea_colon':
        print('DataLoaderMatColon')
        return DataLoaderMatColon(**kwargs)

    raise ValueError('Error! Database type specified wrong (either in config.py or in data_loader.py)')


if __name__ == '__main__':
    dataLoader = get_data_loader()
    dataLoader.files_read_and_save_to_npz('/work/users/mi186veva/data', '/work/users/mi186veva/data_preprocessed/raw')
