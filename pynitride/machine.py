import os
from multiprocessing import cpu_count
from multiprocessing import Pool as _ProcessPool
from multiprocessing.dummy import Pool as _ThreadPool
from operator import itemgetter
from threading import Lock
from contextlib import contextmanager

class Pool():

    @classmethod
    def configure_globalparallel(cls):
        cls.configure()

    @classmethod
    def configure_onlycextparallel(cls):
        cls.configure(globalprocesses=1,globalthreads=1,cextthread=None)

    @classmethod
    def configure(cls, globalthreads=cpu_count()-1,globalprocesses=cpu_count()-1,cextthread=1):
        kwargs={'globalthreads':globalthreads,'globalprocesses':globalprocesses,'cextthread':cextthread}
        if hasattr(cls,'_kwargs'):
            assert kwargs==cls._kwargs, "Pool cannot be reconfigured."
            print("Pool was already configured with the given arguments.")
            return

        cls._kwargs={'globalthreads':globalthreads,'globalprocesses':globalprocesses,'cextthread':cextthread}
        cls._globs={}
        cls._globlck=Lock()
        if cextthread is not None:
            os.environ["OMP_NUM_THREADS"] = str(cextthread)
            os.environ["OPENBLAS_NUM_THREADS"] = str(cextthread)
            os.environ["MKL_NUM_THREADS"] = str(cextthread)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(cextthread)
            os.environ["NUMEXPR_NUM_THREADS"] = str(cextthread)

        if globalprocesses>1:
            cls._procpool=_ProcessPool(processes=globalprocesses,maxtasksperchild=10)
        else: cls._procpool=FakePool()
        if globalthreads>1:
            cls._thrdpool=_ThreadPool(processes=globalprocesses)
        else: cls._thrdpool=FakePool()

    @classmethod
    def process_pool(cls,new=False):
        if not hasattr(cls,'_kwargs'):
            cls.configure()
        elif new:
            cls._refresh_pool()
        return cls._procpool

    @classmethod
    def thread_pool(cls):
        if not hasattr(cls,'_kwargs'):
            cls.configure()
        return cls._thrdpool

    @classmethod
    def _refresh_pool(cls):
        globalprocesses=itemgetter('globalprocesses')(cls._kwargs)
        if globalprocesses>1:
            cls._procpool.close()
            cls._procpool.join()
            cls._procpool=_ProcessPool(processes=globalprocesses,maxtasksperchild=10)
    
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

class FakePool():
    def starmap(self,func,iterable,chunksize=1):
        assert chunksize==1, "Fake pool not implementend for non-unity chunksize"
        return [func(*i) for i in iterable]
    def apply(self,func,args=(),kwds=()):
        return func(*args,**kwds)


_storage={}
_storagepids={}
_storagelock=Lock()
def glob_store(obj):
    with _storagelock:
        key=len(_storage)
        _storage[key]=obj
        _storagepids[key]=os.getpid()
    return key
def glob_read(key):
    return _storage[key]
def glob_remove(key):
    with _storagelock:
        if _storagepids[key]==os.getpid():
            del _storage[key]
            del _storagepids[key]
