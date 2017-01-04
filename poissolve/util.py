
class MultilevelDict():
    def __init__(self,dictionary):
        self._dict=dictionary

    def __getitem__(self, key):
        if isinstance(key, str): key=[key]
        try:
            d=self._dict
            for k in key:
                d=d[k]
            return d
        except:
            raise Exception("Multilevel key error: " + str(key))

    def get(self,param,default=None):
        try:
            return self.__getitem__(param)
        except:
            return default

    def __getattr__(self, item):
        return getattr(self._dict,item)

if __name__=='__main__':
    from runpy import run_path
    run_path('tests/test_util.py',run_name='__main__')

    #from poissolve.tests.test_util import test_multilevel_dict
    #test_multilevel_dict()