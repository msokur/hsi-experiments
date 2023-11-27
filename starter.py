import os
import datetime
import numpy as np
import csv

from archive import config
from provider import get_data_loader
from data_utils.data_loaders.archive.data_loader_base import DataLoader

def cross_validation_starter(root_folder_name, csv_filename=None):
    config.MODEL_NAME_PATHS.append(root_folder_name)

    root_folder = os.path.join(*config.MODEL_NAME_PATHS)
    config.MODEL_NAME = config.get_model_name(config.MODEL_NAME_PATHS)

    if not os.path.exists(root_folder):
        os.mkdir(root_folder)

    data_loader = get_data_loader()
    paths, splits = data_loader.get_paths_and_splits()

    date_ = datetime.datetime.now().strftime("_%d.%m.%Y-%H_%M_%S")

    if csv_filename is None:
        csv_filename = os.path.join(root_folder, root_folder_name + '_stats' + date_ + '.csv')

    for ind, indexes in enumerate(splits):
        model_name = config.MODEL_NAME #config.MODEL_NAME
        if len(indexes) > 1:
            for i in indexes:
                model_name += '_' + str(i)
        else:
            model_name += '_' + str(ind) + '_' + DataLoader.get_name_easy(np.array(paths)[indexes][0])
            # скопировала на всякий случай с сервера, когда-то тут была ошибка
            # config.model_name += '_' + str(indexes[0]) + '_' + np.array(paths)[indexes][0].split("/")[-1].split(".")[0].split('SpecCube')[0]

        
        paths_patch = np.array(paths)[indexes]
        
        except_names = ';'.join([DataLoader.get_name_easy(p) for p in paths_patch])
        
        print('model_name', model_name)
        print('except_names', except_names)
        
        stream = os.popen(f'bash /home/sc.uni-leipzig.de/mi186veva/hsi-experiments/scripts/start_ss_cv_step.sh {model_name} {except_names}')
        output = stream.read()
        print(output)

        #CrossValidator.cross_validation_step(model_name, except_names=)

        for i, path_ in enumerate(paths_patch):

            sensitivity, specificity = 0, 0
            with open(csv_filename, 'a', newline='') as csvfile:  # for full cross_valid and for separate file
                fieldnames = ['time', 'index', 'sensitivity', 'specificity', 'name', 'model_name']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writerow({'time': datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                 'index': str(i),
                                 'sensitivity': str(sensitivity),
                                 'specificity': str(specificity),
                                 'name': path_,
                                 #'model_name': config.MODEL_NAME})
                                 'model_name': model_name})

    return csv_filename

if __name__ == "__main__":

    cross_validation_starter('CVn_3d_inception_v20')


    
    
    