from pynitride.compact.models import GaNHEMT_iMVGS
import pytest

if __name__=="__main__":
    pytest.main(args=[__file__,"-s","--plots"])

def test_ganhemt():
    hemt=GaNHEMT_iMVGS()
    print("hi")