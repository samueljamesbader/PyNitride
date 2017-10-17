import numpy as np
import skvideo.io as sio

def read(filename):
    frames=sio.vreader(filename)
    meta=sio.ffprobe(filename)['video']
    times=np.linspace(0,float(meta['@duration']),int(meta['@nb_frames']))
    return {'times':times, 'frames':frames}
