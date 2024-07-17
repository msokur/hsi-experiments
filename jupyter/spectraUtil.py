import glob as glob
import sys
import os
import inspect

#sys.path.insert(0, '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments') 
#sys.path.insert(0, '/home/sc.uni-leipzig.de/mi186veva/hsi-experiments/data_utils') 
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'data_utils'))

import scaler
import provider

from configuration.parameter import STORAGE_TYPE
import configuration.get_config as config

class SpectraUtil:
    def __init__(self, source_folder, add_standard_scaler=False):
        self.source_folder = source_folder
        self.add_standard_scaler = add_standard_scaler
        
        self.init_scalers()
        
    def init_scalers(self):
        data_storage = provider.get_data_storage(typ=STORAGE_TYPE)
        self.normalizer = scaler.NormalizerScaler(config, self.source_folder, data_storage=data_storage)
        self.standard_transposed_scaler = scaler.StandardScalerTransposed(config, self.source_folder, data_storage=data_storage)
        
        if self.add_standard_scaler:
            self.standard = scaler.StandardScaler(self.source_folder)
            return self.normalizer, self.standard_transposed_scaler, self.standard
        return self.normalizer, self.standard_transposed_scaler
    
    def get_full_X_y(self):
        X, y, _ = self.normalizer.X_y_concatenate()
        return X, y
    
    def get_part_of_X_y(self, X, y, each_Xth_sample=10000):
        return X[::each_Xth_sample], y[::each_Xth_sample]
    
    def get_scaled_X(self, X):
        X_normalized = self.normalizer.scale_X(X)
        X_standard_scaled_T = self.standard_transposed_scaler.scale_X(X)
        
        if self.add_standard_scaler:
            X_standard_scaled = self.standard.scale_X(X)
            return X_normalized, X_standard_scaled_T, X_standard_scaled
        return X_normalized, X_standard_scaled_T
        
    def get_healthy_and_ill_indexes(self, y):
        return y == 0, y == 1
        



'''#++++++++++++++++++++
#get data 1d
paths = glob.glob('/work/users/mi186veva/data_1d/raw/*.npz')
print(len(paths))
X, y = [], []

for path in tqdm(paths):
    data = np.load(path)
    X_, y_ = data['X'], data['y']
    
    X += list(X_)
    y += list(y_)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

#++++++++++++++++++++  1d
#draw distributions histogram

for feature in range(X.shape[-1]):
    print(feature)
    plt.hist(X[:, feature], bins=100)
    plt.show()
    
#++++++++++++++++++++ 1d
X_ = X[::10000]
X_.shape

#++++++++++++++++++++ 1d
###scale

scaler_l2 = preprocessing.Normalizer().fit(X_)
scaler_svn = preprocessing.StandardScaler().fit(X_)
scaler_svn_t = preprocessing.StandardScaler().fit(X_.T)

#++++++++++++++++++++ 1d
###transform

X_l2, X_svn = scaler_l2.transform(X_), scaler_svn.transform(X_)

X_t = scaler_svn_t.transform(X_.T).T

#++++++++++++++++++++ 1d
y_ = y[::10000]
ill = y_ == 1
healthy = y_ == 0
y_.shape'''