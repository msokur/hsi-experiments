from hypercube_data import *
import config
import os
import glob
import cv2
import random
from tqdm import tqdm
import pickle
from sklearn import preprocessing
from augmentator import *
from multiprocessing import Pool

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
        #return image, spectrum_data, np.array(list(np.array(gesund_indexes).T) + list(np.array(ill_indexes).T) + list(np.array(not_certain_indexes).T))
        return image, spectrum_data, np.array(list(np.array(gesund_indexes).T), list(np.array(ill_indexes).T), list(np.array(not_certain_indexes).T))
    else:
        return image, spectrum_data, list(np.array(gesund_indexes).T), list(np.array(ill_indexes).T), []

def save_scaler(all_data, scaler_path):

    scaler = None
    if config.NORMALIZATION_TYPE == config.NORMALIZATION_TYPES['svn']:
        scaler = preprocessing.StandardScaler().fit(all_data)
    elif config.NORMALIZATION_TYPE == config.NORMALIZATION_TYPES['l2_norm']:
        scaler = preprocessing.Normalizer().fit(all_data)

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

def read_data_from_dat(paths=None, not_certain_flag=False, except_indexes=[-1], include_empty=False):
    print('Reading data from .dat files started')
    gesund_data = []
    ill_data = []
    not_certain_data = []

    if paths is None:
        paths = []
        for path_dir in config.DATA_PATHS:
            paths += glob.glob(os.path.join(path_dir, '*.dat'))
    for index, path in tqdm(enumerate(paths)):
        if index not in except_indexes:
            #print(index)
            with open(path, newline='') as filex:
                filename=filex.name

                spectrum_data, _ = Cube_Read(filename, wavearea=config.WAVE_AREA, Firstnm=config.FIRST_NM, Lastnm=config.LAST_NM).cube_matrix()

                mask = cv2.imread(glob.glob(filename + '*.png')[0])[..., ::-1]

                gesund_indexes, ill_indexes, not_certain_indexes = get_masks(mask)

                gesund_patch = spectrum_data[gesund_indexes[0], gesund_indexes[1]]
                #print(gesund_patch.shape)
                ill_patch = spectrum_data[ill_indexes[0], ill_indexes[1]]
                #print(ill_patch.shape)


                if not_certain_flag:
                    not_certain_patch = spectrum_data[not_certain_indexes[0], not_certain_indexes[1]]
                    fill_array = np.zeros(not_certain_patch.shape[0])
                    fill_array.fill(2)
                    not_certain_patch = np.insert(not_certain_patch, not_certain_patch.shape[1], fill_array, axis=1)

                    not_certain_data.append(not_certain_patch)

                if gesund_patch.shape[0] > 0 or include_empty:
                    gesund_patch = np.insert(gesund_patch, gesund_patch.shape[1], np.zeros(gesund_patch.shape[0]), axis=1)
                    gesund_data.append(gesund_patch)

                if ill_patch.shape[0] > 0 or include_empty:
                    ill_patch = np.insert(ill_patch, ill_patch.shape[1], np.ones(ill_patch.shape[0]), axis=1)
                    ill_data.append(ill_patch)
        else:
            print('We are skipping index: ', index)
    print('Reading data from .dat files ended')
    
    return gesund_data, ill_data, not_certain_data, paths

def read_data_from_npy(paths=None, not_certain_flag=False, except_indexes=[-1]):
    print('Reading data from .npy files started')
    gesund_data = []
    ill_data = []
    not_certain_data = []
    paths_res = []

    if paths is None:
        paths = []
        for path_dir in config.NPY_PATHS:
            paths += glob.glob(os.path.join(path_dir, '*.npz'))

    for index, path in tqdm(enumerate(paths)):
        if index not in except_indexes:
            data = np.load(path)
            X, _y = data['X'], data['y']
                        
            y = []
            if len(X.shape) == 3:
                y = [ [_y_] * X.shape[1] for _y_ in _y]
                X = np.concatenate(X, axis=0)
                y = np.concatenate(y, axis=0)
            else:
                y = _y
                
            y = np.array(y)
            
            g = X[np.where(y == 0)]
            gesund_data.append(g)
            i = X[np.where(y == 1)]
            ill_data.append(i)
            n = X[np.where(y == 2)]
            not_certain_data.append(n)
            
            paths_res.append(path)
    
    print('Reading data from .npy files ended')
    
    return gesund_data, ill_data, not_certain_data, paths_res

def split_data(gesund_data, ill_data, not_certain_data, scaler_path, not_certain_flag=False):
    gesund_all = np.concatenate(np.array(gesund_data, dtype=object), axis=0).shape[0]
    ill_all = np.concatenate(np.array(ill_data, dtype=object), axis=0).shape[0]

    gesund_data = [fill_with_weight(i, gesund_all) for i in gesund_data]
    print('----------')
    ill_data = [fill_with_weight(i, ill_all) for i in ill_data]

    gesund_data = list(np.concatenate(np.array(gesund_data, dtype=object), axis=0))                  #label 0
    ill_data = list(np.concatenate(np.array(ill_data, dtype=object), axis=0))     #label 1

    if not_certain_flag: #TODO needs rewirting weights
        not_certain_data_ = not_certain_data.copy()
        not_certain_all = np.concatenate(np.array(not_certain_data, dtype=object), axis=0)
        
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

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('class_weights', class_weight)

    return train, test, class_weight

def get_data(scaler_path, paths=None, 
             return_only_train_dataset = True, 
             not_certain_flag = False, 
             except_indexes=[-1]):
    if scaler_path is None:
        scaler_path = "."
    
    gesund_data, ill_data, not_certain_data = [], [], []
    if config.DATA_LOADER_MODE == 0:
        gesund_data, ill_data, not_certain_data, _ = read_data_from_dat(paths=paths, not_certain_flag=not_certain_flag, except_indexes=except_indexes)
    else:
        gesund_data, ill_data, not_certain_data, _ = read_data_from_npy(paths=paths, not_certain_flag=not_certain_flag, except_indexes=except_indexes)

    train, test, class_weight = split_data(gesund_data, ill_data, not_certain_data, scaler_path, not_certain_flag=not_certain_flag)

    if return_only_train_dataset:
        print('return return')
        return train, test, class_weight
    
    #return all lists
    return [np.array(l) for l in [train, test, gesund_data, ill_data, not_certain_data]], class_weight

def append(lst, X, y, label):
    if len(lst) != 0:
        X += lst
        y += [label] * len(lst)
        
            
def save_npy_from_dat(npy_save_path, dat_paths=None, not_certain_flag=True, except_indexes=[-1]):
    gesund_data, ill_data, not_certain_data, paths = read_data_from_dat(paths=None, not_certain_flag=not_certain_flag, except_indexes=except_indexes)
    
    print('The saving of .npz archives is started')
    
    for it in tqdm(range(len(paths))):
        g, i, n, p = list(gesund_data[it]), list(ill_data[it]), list(not_certain_data[it]), paths[it]
        
        X, y = [], []
        append(g, X, y, 0)
        append(i, X, y, 1)
        if not_certain_flag:
            append(n, X, y, 2)
        
        name = p.split('/')[-1].split('SpecCube')[0]
        np.savez(os.path.join(npy_save_path, name), X=X, y=y)
    
    print('The saving of .npz archives is ended')
    
def save_raw_healthy_and_augmented_ill_in_npy(npy_save_path, augmented_path, dat_paths=None, not_certain_flag=True, except_indexes=[-1], with_raw_ill=False):
    gesund_data, ill_data, not_certain_data, paths = read_data_from_dat(paths=None, not_certain_flag=not_certain_flag, except_indexes=except_indexes, include_empty=True)
    print(len(gesund_data), len(ill_data), len(not_certain_data), len(paths))
    
    print('The saving of raw healthy and augmented ill is started')
    aug_paths = np.array(glob.glob(os.path.join(augmented_path, "*.npz")))
    
    if not os.path.exists(npy_save_path):
        os.mkdir(npy_save_path)
    
    for it, p in tqdm(enumerate(paths)):
        name = p.split('/')[-1].split('SpecCube')[0]
        print(name)
        
        print(aug_paths[np.flatnonzero(np.core.defchararray.find(aug_paths, name) != -1)])
        aug_path = aug_paths[np.flatnonzero(np.core.defchararray.find(aug_paths, name) != -1)]
        print(aug_path)
        if len(aug_path) > 1:
            print(f"Warning! Path {p} has more then one augmented pair!")
        
        if len(aug_path) > 0:
            aug_data = np.load(aug_path[0])
            aug_X, aug_y = aug_data['X'], aug_data['y']
            i = aug_X[np.nonzero(aug_y == 1)]
            if i.shape[0] != 0:
                if with_raw_ill:
                    i = np.concatenate(i[:, :-1, :], axis=0)
                else:
                    i = np.concatenate(i, axis=0)
            
            i = list(i)
            g = list(gesund_data[it])

            X, y = [], []
            append(g, X, y, 0)
            append(i, X, y, 1)
            
            if with_raw_ill:
                i_raw = list(ill_data[it])
                append(i_raw, X, y, 1)
                

            #TODO for now there is no not_certain_data

            np.savez(os.path.join(npy_save_path, name), X=X, y=y)
        else:
            print(f"Warning! Path {p} hasn't augmented pair!")

    print('The saving of raw healthy and augmented ill is ended')
        

def augment(source_paths, destination_paths, normalize_first=False):
    print('Augmentation started')
    for counter, source_path in enumerate(source_paths):
        paths = glob.glob(os.path.join(source_path, "*.npz"))

        for p in tqdm(paths):
            data = np.load(p)
            X, y = data['X'], data['y']
            if normalize_first:
                X[:, :-1] = preprocessing.Normalizer().transform(X[:, :-1])
            name = p.split('/')[-1].split('dat')[0]
            result_X = augment_all(X)  
            destination_path = destination_paths[counter]
            if not os.path.exists(destination_path):
                os.mkdir(destination_path)
            np.savez(os.path.join(destination_path, name), X=result_X, y=y)
    print('Augmentation ended')



if __name__ == '__main__':
    save_raw_healthy_and_augmented_ill_in_npy("/work/users/mi186veva/data_preprocessed/combi_with_raw_ill", "/work/users/mi186veva/data_preprocessed/augmented")
    #save_raw_data()
    #get_data_npy('data_preprocessed//augmented')

    #preprocess('data_preprocessed//augmented')
    #train, test, _ = get_data("./")
    
    #save_npy_from_dat("/work/users/mi186veva/data_preprocessed/raw")
    #data = np.load('/work/users/mi186veva/data_preprocessed/raw/2019_07_12_11_15_49_SpecCube.dat.npz')
    #print(data['gesund_data'].shape)
    #print(data['ill_data'].shape)
    #print(data['not_certain_data'].shape)
    #print(data['path'])

    #augment(["/work/users/mi186veva/data_preprocessed/raw"], ['/work/users/mi186veva/data_preprocessed/augmented_l2_norm'], normalize_first=True)

    #g, i, n, p = read_data_from_npy()
    #Aprint(g.shape, i.shape, n.shape)
    
    #print(train.shape, test.shape)



    #pool = Pool(os.cpu_count())
    #aug = Augmentator(pool)

    '''result = augment_all(train[:1000])

    print(np.array(result).shape)


    print("done")'''