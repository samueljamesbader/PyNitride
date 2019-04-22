"""
Utilities to manage parallelism via multiprocessing.

Importing this module (as is done automatically by importing `pynitride`, configures the environment in accordance
with `config.py` (by default, this enables PyNitride to parallelize with as many processes as CPUs, and prevents numpy/
scipy or other common c-extensions which PyNitride uses from employing any internal parallelization.

Worker pools of both the process and thread variety are available by calling :func:`process_pool` and
:func:`thread_pool` respectively.  A :class:`FakePool` class is provided to mimic either of these without actually
creating more processes or threads.  A :func:`no_parallel` context manager is provided to temporarily make the other
functions use `FakePools` instead of real pools.

Functions such as :func:`glob_store` enable large objects to be stored by a reference so that they don't get pickled
when spawning new processes.  This utility is further enhanced by the :func:`glob_store_attributes` wrapper which can
automatically employ the :func:`glob_store` functions behind the scenes for specified large attributes so that the
object can be safely pickled.
"""
import os
from multiprocessing import cpu_count
from multiprocessing import Pool as _ProcessPool
from multiprocessing.dummy import Pool as _ThreadPool
from operator import itemgetter
from threading import Lock, RLock
from contextlib import contextmanager
from functools import partial,wraps
from pynitride import log
from pynitride import config


###
# Providing the worker pools
###

def process_pool(new=False):
    """ Returns a pool of worker processes.

    If parallelism is enabled in this context, this will be a :py:class:`multiprocessing.pool.Pool`.
    Otherwise, it will be an object which superficially implements the same methods, but runs tasks serially.

    Args:
        new: if returning a real `Pool`, then refresh it first

    Returns:
        an object at least superficially resembling :class:`~multiprocessing.pool.Pool`

    """
    if _no_parallel: return FakePool()
    elif new: _refresh_pool()
    return _procpool

def thread_pool():
    """ Returns a pool of worker threads.

    If parallelism is enabled in this context, this will be a :class:`~multiprocessing.dummy.pool.Pool`.
    Otherwise, it will be an object which superficially implements the same methods, but runs tasks serially.

    Returns:
        an object at least superficially resembling :class:`~multiprocessing.pool.Pool`

    """
    if _no_parallel: return FakePool()
    return _thrdpool

# To close, join, and re-open a process pool
def _refresh_pool():
    global _procpool
    if _no_parallel: return
    if globalprocesses>1:
        _procpool.close()
        _procpool.join()
        _procpool=_ProcessPool(processes=globalprocesses,maxtasksperchild=30)

@contextmanager
def no_parallel():
    """ Context manager to temporarily disable PyNitride-based parallelism."""
    global _no_parallel
    prev=_no_parallel
    _no_parallel=True
    yield
    _no_parallel=prev

def parallel_enabled():
    """ Returns whether parallelism is enabled in this context, see :func:`no_parallel`"""
    return _no_parallel

# Classes to mimic Pool but perform serial tasks
class _FakeAsynchronousResult():
    def wait(self,timeout=None): pass
class FakePool():
    """ Same API as :class:`multiprocessing.pool.Pool` but actually applies serially, no processes/threads."""
    def starmap(self,func,iterable,chunksize=1):
        assert chunksize==1, "Fake pool not implementend for non-unity chunksize"
        return [func(*i) for i in iterable]
    def map(self,func,iterable,chunksize=1):
        assert chunksize==1, "Fake pool not implementend for non-unity chunksize"
        return [func(i) for i in iterable]
    def apply(self,func,args=(),kwds={}):
        return func(*args,**kwds)
    def apply_async(self,func,args=(),kwds={},callback=None,error_callback=None):
        ret=func(*args,**kwds)
        callback(ret)
        return _FakeAsynchronousResult()
    def close(self):
        pass
    def join(self):
        pass

###
# Global readable-writable storage for read-only use with multiprocessing, to avoid pickling large items
###

# key to use for the next object which needs storage
_nextkey=0

# dict to hold all the globals
_storage={}

# pid of the process which stores each value
_storagepids={}

# lock for accessing storage
_storagelock=RLock()

def glob_store(obj):
    """ Add `obj` to the global storage.

    Note: when multiprocessing, be aware that changes made in one process
    do not affect a previously spawned worker process.
    Use `new=True` with :func:`process_pool` in the parent process to incorporate recent updates.

    Args:
        obj: anything.

    Returns:
        a key which can be used in :func:`glob_read` or :func:`glob_write` or :func:`glob_remove`.

    """
    global _nextkey
    with _storagelock:
        key=_nextkey
        _nextkey+=1
        _storage[key]=obj
        _storagepids[key]=os.getpid()
    return key
def glob_read(key):
    """ Returns an object from the global storage.

    Note: when multiprocessing, be aware that changes made in one process
    do not affect a previously spawned worker process.
    Use `new=True` with :func:`process_pool` in the parent process to incorporate recent updates.

    Args:
        key: returned from :func:`glob_store`

    Returns:
        the stored object
    """
    return _storage[key]
def glob_update(key,obj):
    """ Updates an object in the global storage.

    Note: when multiprocessing, be aware that changes made in one process
    do not affect a previously spawned worker process.
    Use `new=True` with :func:`process_pool` in the parent process to incorporate recent updates.

    Args:
        key: returned from :func:`glob_store`
        obj: the new value

    Returns:
        None
    """
    assert key in _storage
    _storage[key]=obj

def glob_remove(key):
    """ Remove an object from the global storage.

    Note: if called a child process other than the one which placed this object on the store,
    this method will not even attempt removal.

    Note: when multiprocessing, be aware that changes made in one process
    do not affect a previously spawned worker process.
    Use `new=True` with :func:`process_pool` in the parent process to incorporate recent updates.

    Args:
        key: returned from :func:`glob_store`

    Returns:
        None
    """
    with _storagelock:
        if _storagepids[key]==os.getpid():
            del _storage[key]
            del _storagepids[key]

def glob_store_attributes(*attrs):
    """ Class decorator to automatically store certain attributes in the glob_store system.

    For example:

    .. code-block:: python

        @globstore_attributes(big_obj)
        class MyClass:
            def __init__(self):
                self.big_obj = np.empty([1e6,1e6])

    Now `MyClass.big_obj` can be get and set like any other parameter, but behind the scenes a property is in place
    such that the MyClass instance does not actually hold a reference to `big_obj` (just the key to retreive it from
    the glob_store system). Thus if an instance of `MyClass` is sent through a multiprocessing function and gets
    pickled, `big_obj` will not be shared through the pickling but instead through process inheritance.

    Note: when subclassing a class which uses this wrapper, make sure that, if the subclass implements an `__init__` or
    `__del__` function, these implementations call the superclass `__init__` or `__del__` functions.

    """

    # wrapper is the function which gets called on the new class
    def wrapper(cls):

        # Grab any __init__ defined for the new class
        def __default_init__(self): super(cls,self).__init__()
        oinit=cls.__dict__.get('__init__',__default_init__)

        # Make an __init__ that runs glob_store for the given attributes and tracks their keys
        @wraps(cls.__init__)
        def __init__(self,*args,**kwargs):
            self._globkeys=self._globkeys if hasattr(self,'_globkeys') else {}
            self._globkeys[cls]={attr:glob_store(None) for attr in attrs}

            # and then calls the original __init__
            oinit(self,*args,**kwargs)

        # Add this new __init__ to the class
        cls.__init__=__init__

        # Make each attribute a property so the actual getting/setting
        # is done via glob_read/glob_update
        def getter(attr,self):
            return glob_read(self._globkeys[cls][attr])
        def setter(attr,self,val):
            return glob_update(self._globkeys[cls][attr],val)
        for attr in attrs:
            setattr(cls,attr,property(partial(getter,attr),partial(setter,attr),doc=''))

        # Grab any __del__ defined for the new class
        def __default_del__(self):
            if hasattr(super(cls,self),'__del__'): super(cls,self).__del__()
        odel=cls.__dict__.get('__del__',__default_del__)

        # Make a __del__ that runs glob_remove for the given attributes
        # after calling the original __del__
        def __del__(self):
            odel(self)
            for k,key in self._globkeys[cls].items():
                try: glob_remove(key)
                except: log("Trouble removing key",level='debug')

        # Add this new __del__ to the class
        cls.__del__=__del__

        return cls
    return wrapper

class Counter():
    def __init__(self, print_every=10, print_message="Count: {}"):
        """ Provides a thread-safe counter.

        Args:
            print_every: every time this many increments has been met or passed, log a message
            print_message: the message to log
        """
        self._count=0
        self._lock=Lock()
        self._print_every=print_every
        self._print_message=print_message

    def increment(self,inc=1):
        """ Increments the counter by `inc`."""
        with self._lock:
            next_milestone=int(self._count/self._print_every)*self._print_every+self._print_every
            self._count+=inc
            if self._count>=next_milestone:
                log(self._print_message.format(self._count))

def raiser(e):
    """ Trivial functional form of the `raise` keyword"""
    raise e


###
# Implementing configuration of parallelism
###

# Read configuration
try:globalthreads=config.getint("parallelism","globalthreads")
except: globalthreads=cpu_count()
try: globalprocesses=config.getint("parallelism","globalprocesses")
except: globalprocesses=cpu_count()
try: cextthread=config.getint("parallelism","cextthread")
except: cextthread=1

# Apply configuration
if cextthread is not None:
    os.environ["OMP_NUM_THREADS"] = str(cextthread)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cextthread)
    os.environ["MKL_NUM_THREADS"] = str(cextthread)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cextthread)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cextthread)

# Create initial pools
if globalprocesses>1:
    _procpool=_ProcessPool(processes=globalprocesses,maxtasksperchild=30)
else: _procpool=FakePool()
if globalthreads>1:
    _thrdpool=_ThreadPool(processes=globalprocesses)
else: _thrdpool=FakePool()

# Start with parallelism on
_no_parallel=False
