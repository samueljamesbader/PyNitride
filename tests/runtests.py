# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:50:17 2017

@author: sam
"""

# Put nothing before this
# because all other lines should be run *after* pytest.main
import pytest
if __name__=='__main__':
    pytest.main()

plots = pytest.mark.skipif(
    not pytest.config.getoption("--plots"),
    reason="need --plots option to run"
)