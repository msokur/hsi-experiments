import os
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob


class Weights: 
    def __init__(self, CONFIG_DATALOADER, Dataloader, weights_filename):
        self.CONFIG_DATALOADER = CONFIG_DATALOADER
        self.Dataloader = Dataloader
        self.weights_filename = weights_filename
        
    def weights_get_from_file(self, root_path):
        weights_path = os.path.join(root_path, self.weights_filename)
        if os.path.isfile(weights_path):
            weights = pickle.load(open(weights_path, 'rb'))
            return weights['weights']
        else:
            raise ValueError("No .weights file was found in the directory, check given path")

    def weights_get_or_save(self, root_path):
        weights_path = os.path.join(root_path, self.weights_filename)

        paths = glob(os.path.join(root_path, '*.npz'))
        y_unique = pickle.load(open(os.path.join(root_path, self.Dataloader.get_labels_filename()), 'rb'))

        quantities = []
        for path in tqdm(paths):
            data = np.load(path)
            X, y = data['X'], data['y']

            quantity = []
            for y_u in y_unique:
                quantity.append(X[y == y_u].shape[0])

            quantities.append(quantity)

        quantities = np.array(quantities)

        sum_ = np.sum(quantities[:, self.CONFIG_DATALOADER["LABELS"]])
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = sum_ / quantities

        weights[np.isinf(weights)] = 0

        data = {
            'paths': paths,
            'weights': weights,
            'sum': sum_,
            'quantities': quantities
        }

        with open(weights_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        return weights

    def weightedData_save(self, root_path, weights):
        paths = glob(os.path.join(root_path, "*.npz"))
        for i, path in tqdm(enumerate(paths)):
            data = np.load(path)
            X, y = data["X"], data["y"]
            weights_ = np.zeros(y.shape)

            for j in np.unique(y):
                weights_[y == j] = weights[i, j]

            data_ = {n: a for n, a in data.items()}
            data_["weights"] = weights_

            np.savez(os.path.join(root_path, self.Dataloader.get_name(path)), **data_)