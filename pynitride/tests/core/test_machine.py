# -*- coding: utf-8 -*-
r""" Tests the global storage """
from pynitride.core.machine import glob_store_attributes, _storage
import pytest
if __name__=="__main__": pytest.main(args=[__file__,'-s'])


@glob_store_attributes('a','b')
class Dad:
    def __init__(self,b):
        self.a='a'
        self.b=b

@glob_store_attributes('c')
class NegligentSon(Dad):
    def __init__(self,c):
        super().__init__(b='b')
        self.c=c

@glob_store_attributes('c')
class CarefulSon(Dad):
    def __init__(self,c):
        super().__init__(b='b')
        self.c=c
    def __del__(self):
        super().__del__()

def test_dad():
    assert len(_storage)==0
    d=Dad(b='b')
    assert (d.a,d.b)==('a','b')
    assert len(_storage)==2
    del d
    assert len(_storage)==0

def test_negligent_son():
    assert len(_storage)==0
    s=NegligentSon(c='c')
    assert (s.a,s.b,s.c)==('a','b','c')
    assert len(_storage)==3
    del s
    assert len(_storage)==0

def test_careful_son():
    assert len(_storage)==0
    s=CarefulSon(c='c')
    assert (s.a,s.b,s.c)==('a','b','c')
    assert len(_storage)==3
    del s
    assert len(_storage)==0

def test_sons():
    assert len(_storage)==0
    s1=CarefulSon(c='c1')
    s2=CarefulSon(c='c2')
    assert (s1.a,s1.b,s1.c)==('a','b','c1')
    assert (s2.a,s2.b,s2.c)==('a','b','c2')
    assert len(_storage)==6
    del s1
    assert len(_storage)==3
    del s2
    assert len(_storage)==0


#if __name__=="__main__":
#    test_dad()
#    test_negligent_son()
#    test_careful_son()
#    test_sons()
