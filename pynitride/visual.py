import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from contextlib import contextmanager
from datetime import datetime
import traceback

_logfile=None
def start_log_file(filename,overwrite=True):
    global _logfile
    if _logfile is not None:
        _logfile.close()
    if overwrite:
        _logfile=open(filename,'w')
    else:
        _logfile=open(filename,'a')


def log(msg,level="info"):
    msg=str(msg)
    if log._levels.index(level)<=log._showlevel:
        print(str(datetime.now())\
            +"     "+"  "*log._depth+msg+"\n",end='',flush=True)
        if _logfile is not None:
            print(str(datetime.now())\
                +"     "+"  "*log._depth+msg+"\n",end='',
                file=_logfile,flush=True)
log._depth=0
log._levels=["error","warning","info","debug","TODO"]
log._showlevel=log._levels.index("info")

def log_fail():
    log("Program failed",level="error")
    log(traceback.format_exc(),level="error")
    

@contextmanager
def sublog(msg,level="info"):
    log(msg,level)
    log._depth+=1
    try:
        yield
    finally:
        log._depth-=1

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).

    From `this StackExchange answer <https://stackoverflow.com/a/16836182/2081118>`_.
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

c = mcolors.ColorConverter().to_rgb
white2red = make_colormap([c('white'),  c('red')])
"""A colormap which goes from white to red, as opposed to the built-in matplotlib 'Reds' colormap which goes from faint pink to red."""
