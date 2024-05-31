import numpy as np
from scipy.spatial.distance import directed_hausdorff
import functools
import multiprocessing as mp
import time
from multiprocessing import Pool


class Hausdorff:

    NUM_BATCH = 20

    def __init__(self):
        pass

    def hausdorff_metric(self, x, y):
        return directed_hausdorff(x, y)

    def hausdorff_metric_dist(self, coordinates, batch_start, batch_size):
        batch_result = np.zeros((batch_size, len(coordinates)))
        for delta in range(batch_size):
            idx = batch_start + delta
            for idy in range(len(coordinates)):
                try:
                    # print(f'{idx} {idy} : {batch_size}')
                    batch_result[delta, idy] = self.hausdorff_metric(x=coordinates[idx], y=coordinates[idy])[0]
                except Exception as e:
                    print(e)
                    print("**")
        return batch_result

    def __update_matrix(self, array, d, batch_start, batch_size, batch_result):
        for delta in range(batch_size):
            idx = batch_start + delta
            for idy in range(len(array)):
                d[idx][idy] = batch_result[delta, idy]
                d[idy][idx] = batch_result[delta, idy]

    def calculate_distance_matrix(self, array):
        d = np.zeros((len(array), len(array)))
        batch_size = len(array) // self.NUM_BATCH
        rest_size = len(array) % self.NUM_BATCH
        start = time.time()
        with Pool(mp.cpu_count()) as p:
            for i in range(self.NUM_BATCH - 1):
                batch_start = i * self.NUM_BATCH
                func = functools.partial(self.__update_matrix, array, d, batch_start, batch_size)
                p.apply_async(self.hausdorff_metric_dist, args=(array, batch_start, batch_size), callback=func)
            rest_start = self.NUM_BATCH * batch_size
            if rest_size > 0:
                func = functools.partial(self.__update_matrix, array, d, rest_start, rest_size)
                p.apply_async(self.hausdorff_metric_dist, args=(array, rest_start, rest_size), callback=func)
            p.close()
            p.join()
        end = time.time()
        # print(d)
        print(end - start)
        return d
