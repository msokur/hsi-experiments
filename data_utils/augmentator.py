import os
from multiprocessing import Pool
import numpy as np
import time
from configuration import get_config as conf

'''class Augmentator():
    def __init__(self, pool, percent=None, aug_range=None, new_rows_per_sample=None):
        self.pool = pool

        if percent is None:
            self.percent = config.AUGMENTATION['percent']
        if aug_range is None:
            self.range = config.AUGMENTATION['range']
        if new_rows_per_sample is None:
            self.new_rows_per_sample = config.AUGMENTATION['new_rows_per_sample']


    def check_main_params(self, percent, aug_range, new_rows_per_sample):
        if percent is None:
            percent = self.percent
        if aug_range is None:
            aug_range = self.range
        if new_rows_per_sample is None:
            new_rows_per_sample = self.new_rows_per_sample

        return percent, aug_range, new_rows_per_sample

    def augment_one(self, row, percent=None, aug_range=None, new_rows_per_sample=None):

        percent, aug_range, new_rows_per_sample = self.check_main_params(percent, aug_range, new_rows_per_sample)

        #print(percent, aug_range, range(new_rows_per_sample))

        result = []
        for j in range(new_rows_per_sample):
            indexes = np.random.choice(len(row), int(len(row) * percent))
            #print(indexes)
            new_row = row.copy()
            for i, el in enumerate(row):

                if i in indexes:
                    new_row[i] = row[i] + ((-aug_range[0] + aug_range[1]) * np.random.normal() + aug_range[0])
            result.append(new_row)

        #print(result)
        return result



    def augment_all(self, data, percent=None, aug_range=None, new_rows_per_sample=None):
        percent, aug_range, new_rows_per_sample = self.check_main_params(percent, aug_range, new_rows_per_sample)

        pool = Pool(os.cpu_count())
        result = []
        for row in tqdm(data):
            r = self.augment_one(row, percent=percent, aug_range=aug_range, new_rows_per_sample=new_rows_per_sample)
            result.append(r)
            #pool.map(self.augment_one, data)

        return result'''


def augment_one(row, row_range=None,
                percent=conf.CONFIG_AUG['percent'],
                aug_range=conf.CONFIG_AUG['range'],
                new_rows_per_sample=conf.CONFIG_AUG['new_rows_per_sample']):
    # print(percent, aug_range, range(new_rows_per_sample))
    if row_range is None:
        if "LAST_NM" in conf.CONFIG_DATALOADER.keys() and "FIRST_NM" in conf.CONFIG_DATALOADER.keys():
            row_range = [0, conf.CONFIG_DATALOADER["LAST_NM"] - conf.CONFIG_DATALOADER["FIRST_NM"]]
        elif "OUTPUT_SIGNATURE_X_FEATURES" in conf.CONFIG_DATALOADER.keys():
            row_range = [0, conf.CONFIG_DATALOADER["OUTPUT_SIGNATURE_X_FEATURES"]]
        else:
            raise ValueError("No spectrum size in Dataloader config!")
    row = np.array(row)
    result = []
    for j in range(new_rows_per_sample):
        indexes = np.random.choice(row[row_range[0]:row_range[1]].shape[0],
                                   int(row[row_range[0]:row_range[1]].shape[0] * percent))
        # print(indexes)
        new_row = row.copy()
        for i, el in enumerate(row):

            if i in indexes:
                # new_row[i] = row[i] + ((-aug_range[0] + aug_range[1]) * np.random.normal()) # + aug_range[0]'''
                new_row[i] = row[i] + np.random.normal(scale=aug_range[1])
        result.append(new_row)

    # print(result)
    return result


def augment_all(data, percent=conf.CONFIG_AUG['percent'],
                aug_range=conf.CONFIG_AUG['range'],
                new_rows_per_sample=conf.CONFIG_AUG['new_rows_per_sample']):
    print('cpu count', os.cpu_count())

    start = time.time()
    pool = Pool(int(os.cpu_count()))
    result = []
    # for row in tqdm(data):
    # r = augment_one(row, percent=percent, aug_range=aug_range, new_rows_per_sample=new_rows_per_sample)

    result = pool.map(augment_one, data)
    pool.close()
    pool.join()

    end = time.time()
    print('time', end - start)

    return result


if __name__ == '__main__':
    # pool = Pool(os.cpu_count())
    # aug = Augmentator(pool)
    d = [[1, 2, 3, 0], [2, 3, 4, 0]]

    # p = Process(target=c, args=(d[0][0]))
    # p.start()
    # p.join()

    result_ = augment_all(d)
    print(np.array(result_).shape)
    print(np.concatenate(np.array(result_), axis=0).shape)
