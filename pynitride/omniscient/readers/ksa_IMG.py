import numpy as np

def read(filename,width=328,height=246):
    dtype = np.dtype('<u2')
    with open(filename, 'rb') as fid:
        data = np.fromfile(fid, dtype)
    meta = data.shape[0]-width*height
    image=data[meta:].reshape((height,width))
    return {'image':image, '_metadata': data[:meta]}