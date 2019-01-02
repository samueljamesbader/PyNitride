import os
from multiprocessing import cpu_count
from multiprocessing import Pool as _ProcessPool
from multiprocessing.dummy import Pool as _ThreadPool

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
        if cextthread is not None:
            os.environ["OMP_NUM_THREADS"] = str(cextthread)
            os.environ["OPENBLAS_NUM_THREADS"] = str(cextthread)
            os.environ["MKL_NUM_THREADS"] = str(cextthread)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(cextthread)
            os.environ["NUMEXPR_NUM_THREADS"] = str(cextthread)

        if globalprocesses>1:
            cls._procpool=_ProcessPool(processes=globalprocesses)
        else: cls._procpool=FakePool()
        if globalthreads>1:
            cls._thrdpool=_ThreadPool(processes=globalprocesses)
        else: cls._thrdpool=FakePool()

    @classmethod
    def process_pool(cls):
        if not hasattr(cls,'_kwargs'):
            cls.configure()
        return cls._procpool

    @classmethod
    def thread_pool(cls):
        if not hasattr(cls,'_kwargs'):
            cls.configure()
        return cls._thrdpool

class FakePool():
    def starmap(self,func,iterable,chunksize=1):
        assert chunksize==1, "Fake pool not implementend for non-unity chunksize"
        return [func(*i) for i in iterable]
    def apply(self,func,args=(),kwds=()):
        return func(*args,**kwds)
