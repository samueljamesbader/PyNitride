# -*- coding: utf-8 -*-
r""" Tests the parameter and units related utilities in :py:mod:`pynitride.paramdb`."""
import numpy as np
from pynitride.paramdb import ParamDB
import pytest

if __name__=="__main__": pytest.main(args=[__file__,'-s'])

actual_hbar=.658211942
actual_e=1

def test_quantity_neu():
    Q_=ParamDB(units='neu').quantity

    # Single works
    hbar=Q_("hbar")
    assert not hasattr(hbar,"units")
    assert np.isclose(hbar,actual_hbar)

    # Comma-string works
    assert True not in [hasattr(a,"units") for a in Q_("hbar,e")]
    assert np.allclose(Q_("hbar,e"),[actual_hbar,actual_e])

    # Straight through works
    hbar=Q_(1,"hbar")
    assert not hasattr(hbar,"units")
    assert np.isclose(hbar,actual_hbar)

    # Doesn't mess with non-numeric arrays
    assert Q_(["eps","hbar"])==["eps","hbar"]

def test_quantity_pint():
    Q_=ParamDB(units='Pint').quantity

    # Single works
    hbar=Q_("hbar")
    assert hbar.units == ParamDB._ureg.hbar
    assert hbar.to_base_units().units==ParamDB._ureg("eV_fs2_per_nm2 * nanometer ** 2 / femtosecond")
    assert np.isclose(hbar.to_base_units().magnitude,actual_hbar)

    # Comma-string works
    assert False not in [hasattr(a,"units") for a in Q_("hbar,e")]
    assert False not in [(a==1*b) for a,b in zip(Q_("hbar,e"),[ParamDB._ureg.hbar,ParamDB._ureg.e])]

    # Straight through works
    hbar=Q_(1,"hbar")
    assert hbar.units==ParamDB._ureg.hbar
    assert hbar==1*ParamDB._ureg.hbar

    # Doesn't mess with non-numeric arrays
    assert Q_(["eps","hbar"])==["eps","hbar"]
