from tests.util.ctest_cython_loops import ctest_dimsimple
import pytest

if __name__ == "__main__":
    pytest.main(args=[__file__,'-s'])


def test_dimsimple():
    ctest_dimsimple()



