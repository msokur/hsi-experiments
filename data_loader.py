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

def read_data_from_dat(paths=None, not_certain_flag=False, except_indexes=[-1]):

    gesund_data = []
    ill_data = []
    not_certain_data = []

    if paths is None:
        for path_dir in config.DATA_PATHS:
            paths = glob.glob(os.path.join(path_dir, '*.dat'))
    for index, path in tqdm(enumerate(paths)):
        if index not in except_indexes:
            print(index)
            with open(path, newline='') as filex:
                filename=filex.name

                spectrum_data, _ = Cube_Read(filename, wavearea=config.WAVE_AREA, Firstnm=config.FIRST_NM, Lastnm=config.LAST_NM).cube_matrix()

                mask = cv2.imread(glob.glob(filename + '*.png')[0])[..., ::-1]

                gesund_indexes, ill_indexes, not_certain_indexes = get_masks(mask)

                gesund_patch = spectrum_data[gesund_indexes[0], gesund_indexes[1]]
                print(gesund_patch.shape)
                ill_patch = spectrum_data[ill_indexes[0], ill_indexes[1]]
                print(ill_patch.shape)


                if not_certain_flag:
                    not_certain_patch = spectrum_data[not_certain_indexes[0], not_certain_indexes[1]]
                    fill_array = np.zeros(not_certain_patch.shape[0])
                    fill_array.fill(2)
                    not_certain_patch = np.insert(not_certain_patch, not_certain_patch.shape[1], fill_array, axis=1)

                    not_certain_data.append(not_certain_patch)

                if gesund_patch.shape[0] > 0:
                    gesund_patch = np.insert(gesund_patch, gesund_patch.shape[1], np.zeros(gesund_patch.shape[0]), axis=1)
                    gesund_data.append(gesund_patch)

                if ill_patch.shape[0] > 0:
                    ill_patch = np.insert(ill_patch, ill_patch.shape[1], np.ones(ill_patch.shape[0]), axis=1)
                    ill_data.append(ill_patch)
        else:
            print('We are skipping index: ', index)
    
    return gesund_data, ill_data, not_certain_data, paths


def get_data(scaler_path, paths=None, 
             return_only_train_dataset = True, 
             not_certain_flag = False, 
             except_indexes=[-1]):
    if scaler_path is None:
        scaler_path = "."
   

    '''print(except_indexes)
    except_indexes = list(except_indexes)
    except_indexes.append(35)
    print(except_indexes)'''

    gesund_data, ill_data, not_certain_data, _ = read_data_from_dat(paths=paths, not_certain_flag=not_certain_flag, except_indexes=except_indexes)
    
    gesund_all = np.concatenate(np.array(gesund_data), axis=0).shape[0]
    ill_all = np.concatenate(np.array(ill_data), axis=0).shape[0]

    gesund_data = [fill_with_weight(i, gesund_all) for i in gesund_data]
    print('----------')
    ill_data = [fill_with_weight(i, ill_all) for i in ill_data]

    gesund_data = list(np.concatenate(np.array(gesund_data), axis=0))                  #label 0
    ill_data = list(np.concatenate(np.array(ill_data), axis=0))     #label 1

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

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('class_weights', class_weight)

    if return_only_train_dataset:
        print('return return')
        return train, test, class_weight
    
    #return all lists
    return [np.array(l) for l in [train, test, gesund_data, ill_data, not_certain_data]], class_weight

def save_npy_from_dat(npy_save_path, dat_paths=None, not_certain_flag=True, except_indexes=[-1]):
    gesund_data, ill_data, not_certain_data, paths = read_data_from_dat(paths=None, not_certain_flag=not_certain_flag, except_indexes=[-1])

    print(np.array(gesund_data, dtype=object).shape, len(paths))

    for g, i, n, p, it in zip(gesund_data, ill_data, not_certain_data, paths, range(len(paths))):
        print(os.path.join(npy_save_path, str(it)))
        name = p.split('/')[-1]
        np.savez(os.path.join(npy_save_path, name), gesund_data=g, ill_data=i, not_certain_data=n, path=p)


def augment(source_path, destiation_path):
    paths = glob.glob(os.path.join(source_path, "*.npz"))

    for p in tqdm(paths[:1]):
        data = np.load(p)
        g, i, n, pth = data['gesund_data'], data['ill_data'], data['not_certain_data'], str(data['path'])
        print(g.shape, i.shape, n.shape, str(pth))
        name = pth.split('/')[-1]
        result_g = augment_all(g) #TODO  добавить номера примеров внутрь, чтобы можно было потом отсортировать какой пример куда относится 
        result_i = augment_all(i)
        result_n = augment_all(n)
        np.savez(os.path.join(destiation_path, name), 
                    gesund_data=result_g, 
                    ill_data=result_i, 
                    not_certain_data=result_n, path=p)


'''def save_class_weights(path, obj):
    with open(os.path.join(path, '.class_weights'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_class_weights(path):
    with open(os.path.join(path, '.class_weights'), 'rb') as f:
        return pickle.load(f)

def get_data_npy(folder, train_part=-1, test_part=-1, all=True):
    train_paths = glob.glob(os.path.join(folder, 'train', '*.npy'))
    test_paths = glob.glob(os.path.join(folder, 'test', '*.npy'))
    class_weights = load_class_weights(folder)

    train = []
    for train_path in train_paths[train_part:train_part+1]:
        train.append(np.load(train_path))
    if all:
        for train_path in train_paths:
            train.append(np.load(train_path))

    test = []
    for test_path in test_paths[test_part:test_part+1]:
        test.append(np.load(test_path))
    if all:
        for test_path in test_paths:
            test.append(np.load(test_path))


    if len(train) > 0:
        print(np.array(train).shape)
        train = np.concatenate(np.array(train), axis=0)

    if len(test) > 0:
        test = np.concatenate(np.array(test), axis=0)

    print(np.array(train).shape, np.array(test).shape, class_weights)

    return train, test, class_weights

def save_patches(arr, folder, folder_2, concat=False):
    if concat:

        arr = np.concatenate(np.array(arr), axis=0)
    existing_paths = glob.glob(os.path.join(folder, folder_2, "*.npy"))
    parts = config.AUGMENTATION['new_rows_per_sample']
    patches = np.array_split(arr, parts)

    for i, patch in enumerate(patches):
        print(patch.shape)
        np.save(os.path.join(folder, folder_2, 'part' + str(len(existing_paths) + i)), patch)

def save_raw_data():
    train, test, class_weights = get_data(None)
    #print(train.shape, test.shape)
    save_class_weights('data_preprocessed/augmented', class_weights)
    save_class_weights('data_preprocessed/raw', class_weights)

    save_patches(train, 'data_preprocessed//raw', 'train')

    #np.save('data_prepsocessed/raw/train/part0', train[:int(train.shape[0] / 2)])
    #np.save('data_prepsocessed/raw/train/part1', train[int(train.shape[0] / 2):])
    np.save('data_preprocessed/raw/test/test', test)



def train_generator(folder='data_preprocessed//raw', part=-1):
    train, _, _ = get_data_npy(folder, train_part=part, all=False)
    for t in train:
        yield t

def test_generator(folder='data_preprocessed//raw', part=-1):
    _, test, _ = get_data_npy(folder, test_part=part, all=False)
    for t in test:
        yield t


def preprocess(save_folder):
    #train, test, class_weights = get_data_npy('data_preprocessed//raw')

    print('Augment train set start')

    paths = glob.glob('data_preprocessed//raw//train//*.npy')
    for i in tqdm(range(1)):
        result_train = augment_all(train_generator(part=i))
        save_patches(result_train, save_folder, 'train', concat=True)

    del result_train

    print('Augment test set start')
    #result_test = augment_all(test_generator(part=0))
    #save_patches(result_test, save_folder, 'test', concat=True)'''


if __name__ == '__main__':
    #save_raw_data()
    #get_data_npy('data_preprocessed//augmented')

    #preprocess('data_preprocessed//augmented')
    #train, test, _ = get_data("./")
    
    #save_npy_from_dat("data_preprocessed/raw")
    data = np.load('data_preprocessed/raw/2019_07_12_11_15_49_SpecCube.dat.npz')
    print(data['gesund_data'].shape)
    print(data['ill_data'].shape)
    print(data['not_certain_data'].shape)
    print(data['path'])

    augment('data_preprocessed/raw', 'data_preprocessed/augmented')


    
    #print(train.shape, test.shape)



    #pool = Pool(os.cpu_count())
    #aug = Augmentator(pool)

    '''result = augment_all(train[:1000])

    print(np.array(result).shape)


    print("done")'''