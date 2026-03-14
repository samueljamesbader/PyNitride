import os
from contextlib import contextmanager
from datetime import datetime
import traceback

# Name of the logfile for a run
_logfile=None

def start_log_file(filename,overwrite=True):
    """ Establishes a log file to direct subsequent messages

    Args:
        filename: path of the log file
        overwrite: whether to overwrite or append
    """
    global _logfile
    if _logfile is not None:
        _logfile.close()
    if overwrite:
        _logfile=open(filename,'w')
    else:
        _logfile=open(filename,'a')


def log(msg,level="info"):
    """ Logs a message if it is severe enough.

    For info on the severity setting, see :func:`set_level`

    Args:
        msg: the string to log
        level: the severity of the message, one of
            "error", "warning", "info", "debug", "TODO"
    """
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

def set_level(level=None):
    """ Sets the minimum severity for a message to be printed.

    The :func:`log` function will ignore messages below the set level.

    Args:
        level: the severity of the message, one of
            "error", "warning", "info", "debug", "TODO".
            Or `None`, to use the default from `config.ini`
    """
    if level is None:
        level=os.environ.get("PYNITRIDE_LOGLEVEL", "info")
    else:
        log("Log level: "+level, level="info")
    log._showlevel=log._levels.index(level)
set_level()

def log_fail():
    """ Convenience function which logs the most recent traceback. """
    log("Program failed",level="error")
    log(traceback.format_exc(),level="error")
    

@contextmanager
def sublog(msg,level="info"):
    """ Context manager to log a message and temporarily indent further messages

    Args:
        msg: the "heading" message
        level: the severity, see :func:`log`
    """
    log(msg,level)
    log._depth+=1
    try:
        yield
    finally:
        log._depth-=1
