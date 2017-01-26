
import pytest
from poissolve.util import MultilevelDict
if __name__=='__main__':
    pytest.main(args=[__file__])

def test_multilevel_dict():
    m=MultilevelDict({'GaN':{'dopants':{'Si':{'type':'donor'}},'eps':10.6}}['GaN'])
    print(m['eps'])
    #assert 0