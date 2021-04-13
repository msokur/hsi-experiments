import os
from multiprocessing import Pool
from multiprocessing import Process
import config
import numpy as np
import random
from tqdm import tqdm
import time


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


def augment_one(row, row_range=[0, config.LAST_NM-config.FIRST_NM], percent=config.AUGMENTATION['percent'],
                     aug_range=config.AUGMENTATION['range'],
                     new_rows_per_sample=config.AUGMENTATION['new_rows_per_sample']):

    #print(percent, aug_range, range(new_rows_per_sample))
    row = np.array(row)
    result = []
    for j in range(new_rows_per_sample):
        indexes = np.random.choice(row[row_range[0]:row_range[1]].shape[0], int(row[row_range[0]:row_range[1]].shape[0] * percent))
        #print(indexes)
        new_row = row.copy()
        for i, el in enumerate(row):

            if i in indexes:
                #new_row[i] = row[i] + ((-aug_range[0] + aug_range[1]) * np.random.normal()) # + aug_range[0]'''
                new_row[i] = row[i] + np.random.normal(scale=aug_range[1])
        result.append(new_row)


    #print(result)
    return result


def augment_all(data, percent=config.AUGMENTATION['percent'],
                      aug_range=config.AUGMENTATION['range'],
                      new_rows_per_sample=config.AUGMENTATION['new_rows_per_sample']):
    print('cpu count', os.cpu_count())

    start = time.time()
    pool = Pool(int(os.cpu_count() / 2))
    result = []
    #for row in tqdm(data):
    #r = augment_one(row, percent=percent, aug_range=aug_range, new_rows_per_sample=new_rows_per_sample)

    result = pool.map(augment_one, data)
    pool.close()
    pool.join()

    end = time.time()
    print('time', end - start)

    return result

if __name__ == '__main__':
    #pool = Pool(os.cpu_count())
    #aug = Augmentator(pool)
    d = [[1, 2, 3, 0], [2, 3, 4, 0]]

    #p = Process(target=c, args=(d[0][0]))
    #p.start()
    #p.join()

    result = augment_all(d)
    print(np.array(result).shape)
    print(np.concatenate(np.array(result), axis=0).shape)

