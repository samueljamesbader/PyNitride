import numpy as np


def read(filename):
    with open(filename) as f:
        data=np.array(np.vstack([[float(x.strip()) for x in l.split()] for l in f]))
        return {'wavelength': data[:,0], 'counts': data[:,1]}
