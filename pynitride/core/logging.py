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
