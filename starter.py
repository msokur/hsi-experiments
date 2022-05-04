import os
import config
from provider import get_data_loader
import datetime

def cross_validation_starter(root_folder_name, csv_filename=None):
    model_name_paths = config.MODEL_NAME_PATHS.copy()
    model_name_paths.append(root_folder_name)

    root_folder = os.path.join(*model_name_paths)
    #model_name = config.get_model_name(model_name_paths)
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)

    data_loader = get_data_loader()
    paths, splits = data_loader.get_paths_and_splits()

    date_ = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")

    if csv_filename is None:
        csv_filename = os.path.join(root_folder, root_folder_name + '_stats' + date_ + '.csv')
    csv_filename = os.path.join(root_folder, csv_filename)

    for ind, indexes in enumerate(splits):

        self.cross_validation_step(csv_filename, ind, indexes, paths)

    return csv_filename

if __name__ == "__main__":

    cross_validation_starter('CV_3d_starter')


    
    
    