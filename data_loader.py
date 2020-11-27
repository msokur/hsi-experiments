from hypercube_data import *
import config
import os
import glob
import cv2
import random
from tqdm import tqdm
import pickle
from sklearn import preprocessing



def get_masks(mask):
    gesund_indexes = np.where((mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 255)) #blue
    not_certain_indexes = np.where((mask[:, :, 0] == 255) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0)) #red
    ill_indexes = np.where((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)) #yellow

    return gesund_indexes, ill_indexes, not_certain_indexes

def get_data_for_showing(name, path, with_not_certain=False):
    spectrum_data, _ = Cube_Read(os.path.join(path, name), wavearea=config.WAVE_AREA, Firstnm=config.FIRST_NM, Lastnm=config.LAST_NM).cube_matrix()
    image_path = glob.glob(os.path.join(path, name + "*.png"))[0]
    image = cv2.imread(image_path)
    gesund_indexes, ill_indexes, not_certain_indexes = get_masks(image[..., ::-1])

    if with_not_certain:
        return image, spectrum_data, np.array(list(np.array(gesund_indexes).T) + list(np.array(ill_indexes).T) + list(np.array(not_certain_indexes).T))
    else:
        return image, spectrum_data, np.array(list(np.array(gesund_indexes).T) + list(np.array(ill_indexes).T))

def save_scaler(all_data, scaler_path):
    scaler = preprocessing.StandardScaler().fit(all_data)

    #TODO realize scaler name that contains run_name
    pickle.dump(scaler, open(os.path.join(scaler_path, config.SCALER_FILE_NAME), 'wb'))

    return scaler

def restore_scaler(scaler_path):
    return pickle.load(open(os.path.join(scaler_path, config.SCALER_FILE_NAME), 'rb'))

def fill_with_weight(arr, total):
    arr = np.array(arr)
    if arr.shape[0] > 0:
        fill_array = np.zeros(arr.shape[0])
        fill_array.fill(total / arr.shape[0])
        #print(total/arr.shape[0])
        arr = np.insert(arr, arr.shape[1], fill_array, axis=1)

    return arr

def get_data(scaler_path, only_train_dataset = True, not_certain_flag = False):
    gesund_data = []
    ill_data = []
    not_certain_data = []

    for path_dir in config.DATA_PATHS:
        for path in tqdm(glob.glob(os.path.join(path_dir, '*.dat'))): 
            with open(path, newline='') as filex:
                filename=filex.name

                spectrum_data, _ = Cube_Read(filename, wavearea=config.WAVE_AREA, Firstnm=config.FIRST_NM, Lastnm=config.LAST_NM).cube_matrix()

                mask = cv2.imread(glob.glob(filename + '*.png')[0])[..., ::-1]

                gesund_indexes, ill_indexes, not_certain_indexes = get_masks(mask)

                gesund_patch = spectrum_data[gesund_indexes[0], gesund_indexes[1]]
                gesund_patch = np.insert(gesund_patch, gesund_patch.shape[1], np.zeros(gesund_patch.shape[0]), axis=1)
                ill_patch = spectrum_data[ill_indexes[0], ill_indexes[1]]
                

                if not_certain_flag:
                    not_certain_patch = spectrum_data[not_certain_indexes[0], not_certain_indexes[1]]
                    fill_array = np.zeros(not_certain_patch.shape[0])
                    fill_array.fill(2)
                    not_certain_patch = np.insert(not_certain_patch, not_certain_patch.shape[1], fill_array, axis=1)

                    not_certain_data.append(not_certain_patch)
                
                gesund_data.append(gesund_patch)
                if ill_patch.shape[0] > 0:
                    ill_patch = np.insert(ill_patch, ill_patch.shape[1], np.ones(ill_patch.shape[0]), axis=1)
                    ill_data.append(ill_patch)

    gesund_all = np.concatenate(np.array(gesund_data), axis=0).shape[0]
    ill_all = np.concatenate(np.array(ill_data), axis=0).shape[0]

    gesund_data = [fill_with_weight(i, gesund_all) for i in gesund_data]
    print('----------')
    ill_data = [fill_with_weight(i, ill_all) for i in ill_data]

    gesund_data = list(np.concatenate(np.array(gesund_data), axis=0))                  #label 0
    ill_data =  list(np.concatenate(np.array(ill_data), axis=0))     #label 1

    if not_certain_flag: #TODO needs rewirting weights
        not_certain_data_ = not_certain_data.copy()
        not_certain_all = np.concatenate(np.array(not_certain_data), axis=0)
        
        not_certain_data = list(not_certain_all)
        print(gesund_data.shape, ill_data.shape, not_certain_data.shape)
    else:
        print(np.array(gesund_data).shape, np.array(ill_data).shape)

    random.shuffle(gesund_data)
    random.shuffle(ill_data)
    random.shuffle(not_certain_data)

    train = []
    test = []

    split_gesund = int(config.SPLIT_FACTOR * len(gesund_data))
    split_ill = int(config.SPLIT_FACTOR * len(ill_data))
    split_not_certain = int(config.SPLIT_FACTOR * len(not_certain_data))

    print('length of input lists (gesund, ill, not_certain)', len(gesund_data), len(ill_data), len(not_certain_data))
    print('split factor', split_gesund, split_ill, split_not_certain)

    if len(gesund_data) > 0:
        train += gesund_data[:split_gesund]
        test += gesund_data[split_gesund:]
    if len(ill_data) > 0:
        train +=ill_data[:split_ill]
        test += ill_data[split_ill:]
    if len(not_certain_data) > 0 and not_certain_flag:
        train += not_certain_data[:split_not_certain]
        test += not_certain_data[split_not_certain:]

    random.shuffle(train)
    random.shuffle(test)

    train = np.array(train)
    test = np.array(test)

    print('train, test length', train.shape, test.shape)

    scaler = save_scaler(train[:, :-2], scaler_path)
    train[:, :-2] = scaler.transform(train[:, :-2])
    test[:, :-2] = scaler.transform(test[:, :-2])

    neg = len(gesund_data)
    pos = len(ill_data)
    total = neg + pos
    
    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0

    class_weight = {0: weight_for_0, 1: 10 * weight_for_1}

    print('class_weights', class_weight)

    if only_train_dataset:
        return train, test, class_weight
    
    #return all lists
    return [np.array(l) for l in [train, test, gesund_data, ill_data, not_certain_data]], class_weight

if __name__ == '__main__':
    train, test, gesund_data, ill_data, not_certain_data = get_data("./", only_train_dataset=False)

    print("done")