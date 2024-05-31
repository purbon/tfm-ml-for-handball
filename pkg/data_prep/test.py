from pprint import pprint

import h5py

if __name__ == '__main__':

    filepath = "handball.h5"
    f = h5py.File(filepath, 'r')
    keys = [key for key in f.keys()]
    pprint(keys)
