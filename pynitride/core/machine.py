import os
from multiprocessing import cpu_count
from multiprocessing import Pool as _ProcessPool
from multiprocessing.dummy import Pool as _ThreadPool
from operator import itemgetter
from threading import Lock, RLock
from contextlib import contextmanager
from functools import partial,wraps
from pynitride import log
from configparser import ConfigParser
from pynitride import ROOT_DIR

class Pool():

    @classmethod
    def configure(cls):
        cp=ConfigParser()
        cp.read(os.path.join(ROOT_DIR,"config.ini"))

        globalthreads=cp.getint("parallelism","globalthreads")
        globalprocesses=cp.getint("parallelism","globalprocesses")
        cextthread=cp.getint("parallelism","cextthread")

        kwargs={'globalthreads':globalthreads,
                'globalprocesses':globalprocesses,'cextthread':cextthread}
        if hasattr(cls,'_kwargs'):
            assert kwargs==cls._kwargs, "Pool cannot be reconfigured."
            log("Pool was already configured with the given arguments.")
            return

        cls._kwargs={'globalthreads':globalthreads,
                'globalprocesses':globalprocesses,'cextthread':cextthread}
        cls._globs={}
        cls._globlck=RLock()
        if cextthread is not None:
            os.environ["OMP_NUM_THREADS"] = str(cextthread)
            os.environ["OPENBLAS_NUM_THREADS"] = str(cextthread)
            os.environ["MKL_NUM_THREADS"] = str(cextthread)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(cextthread)
            os.environ["NUMEXPR_NUM_THREADS"] = str(cextthread)

        if globalprocesses>1:
            cls._procpool=_ProcessPool(processes=globalprocesses,maxtasksperchild=30)
        else: cls._procpool=FakePool()
        if globalthreads>1:
            cls._thrdpool=_ThreadPool(processes=globalprocesses)
        else: cls._thrdpool=FakePool()

    _no_parallel=False

    @classmethod
    def process_pool(cls,new=False):
        if cls._no_parallel: return FakePool()
        if not hasattr(cls,'_kwargs'):
            cls.configure()
        elif new:
            cls._refresh_pool()
        return cls._procpool

    @classmethod
    def thread_pool(cls):
        if cls._no_parallel: return FakePool()
        if not hasattr(cls,'_kwargs'):
            cls.configure()
        return cls._thrdpool

    @classmethod
    def _refresh_pool(cls):
        if cls._no_parallel: return
        globalprocesses=itemgetter('globalprocesses')(cls._kwargs)
        if globalprocesses>1:
            cls._procpool.close()
            cls._procpool.join()
            cls._procpool=_ProcessPool(processes=globalprocesses,maxtasksperchild=30)

    @classmethod
    @contextmanager
    def no_parallel(cls):
        prev=cls._no_parallel
        cls._no_parallel=True
        yield
        cls._no_parallel=prev

    
    @classmethod
    @contextmanager
    def reference(cls,*args):
        if not hasattr(cls,'_kwargs'):
            cls.configure()
        with cls._globlck:
            n=range(len(cls._globs),len(cls._globs)+len(args))
            for i,ni in enumerate(n):
                cls._globs[ni]=args[i]

        #cls._refresh_pool()
        if len(args)==1: yield list(n)[0]
        else: yield list(n)


        with cls._globlck:
            for ni in n:
                del cls._globs[ni]
    @classmethod
    def dereference(cls,*args):
        return itemgetter(*args)(cls._globs)
class FakeAsynchronousResult():
    def wait(self,timeout=None): pass
class FakePool():
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
        return FakeAsynchronousResult()
    def close(self):
        pass
    def join(self):
        pass

_nextkey=0
_storage={}
_storagepids={}
_storagelock=RLock()
def glob_store(obj):
    global _nextkey
    with _storagelock:
        key=_nextkey
        _nextkey+=1
        _storage[key]=obj
        _storagepids[key]=os.getpid()
    return key
def glob_read(key):
    return _storage[key]
def glob_update(key,obj):
    assert key in _storage
    _storage[key]=obj
def glob_remove(key):
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
        self._count=0
        self._lock=Lock()
        self._print_every=print_every
        self._print_message=print_message
    def increment(self,inc=1):
        with self._lock:
            next_milestone=int(self._count/self._print_every)*self._print_every+self._print_every
            self._count+=inc
            if self._count>=next_milestone:
                log(self._print_message.format(self._count))

def raiser(e): raise e
