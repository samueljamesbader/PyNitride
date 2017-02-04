r""" Test the :py:mod:`~pynitride.poissolve.mesh` module."""
import numpy as np
from pynitride.poissolve.mesh import ConstantFunction
import pytest

if __name__=="__main__": pytest.main(args=[__file__,'-s'])


#def test_constant_function(trivialmesh):
#    r""" Make sure that a :py:func:`~.pynitride.poissolve.mesh.ConstantFunction` can't be assigned a non-constant value.
#    """
#
#    # scalar mid function
#    cf=ConstantFunction(trivialmesh,1,pos='mid')
#    assert np.allclose(cf,[1]*len(trivialmesh.zm)), "ConstantFunction on mid mesh not properly created"
#
#    # scalar point function
#    cf=ConstantFunction(trivialmesh,1,pos='point')
#    assert np.allclose(cf,[1]*len(trivialmesh.zp)), "ConstantFunction on point mesh not properly created"
#
#    with pytest.raises(Exception,message="Scalar ConstantFunction allowed non-constant assignment"):
#        cf[1]=2
#    assert np.allclose(cf,[1]*len(trivialmesh.zp))
#
#    cf[:]=3
#    assert np.allclose(cf,[3]*len(trivialmesh.zp)), "Scalar ConstantFunction failed constant assignment"
#
#    with pytest.raises(Exception,message="Scalar ConstantFunction allowed non-constant assignment"):
#        cf[:]=[4,5]
#    assert np.allclose(cf,[3]*len(trivialmesh.zp))
#
#    with pytest.raises(Exception,message="Scalar ConstantFunction allowed non-constant assignment"):
#        cf[:]=np.arange(len(trivialmesh.zp))
#
#    cf2=ConstantFunction(trivialmesh,6,pos='point')
#    with pytest.raises(Exception,message="Scalar ConstantFunction allowed non-constant assignment by ConstantFunction"):
#        cf[1]=cf2
#    cf[:]=cf2
#    assert np.allclose(cf,cf2), "Scalar ConstantFunction failed assigment by ConstantFuntion"
#
#    # vector point function
#    cf=ConstantFunction(trivialmesh,[3,4])
#    assert np.allclose(cf,np.transpose([[3,4]]*len(trivialmesh.zp))), "ConstantFunction on point mesh not properly created"
#
#    with pytest.raises(Exception,message="Vector ConstantFunction allowed non-constant assignment"):
#        cf[1]=[5,6]
#    assert np.allclose(cf,np.transpose([[3,4]]*len(trivialmesh.zp)))

#    cf[:,:]=[7,8]
#    assert np.allclose(cf,np.transpose([[7,8]]*len(trivialmesh.zp))), "Vector ConstantFunction failed constant assignment"
#    cf[:]=[7,8]
#    assert np.allclose(cf,np.transpose([[7,8]]*len(trivialmesh.zp))), "Vector ConstantFunction failed constant assignment"
#
#    with pytest.raises(Exception,message="Vector ConstantFunction allowed non-constant assignment"):
#        cf[:]=[[9,10],[11,12]]
#    assert np.allclose(cf,np.transpose([[7,8]]*len(trivialmesh.zp)))
#
#    with pytest.raises(Exception,message="Vector ConstantFunction allowed non-constant assignment"):
#        cf[:,]=np.reshape(np.arange(len(2*trivialmesh.zp)),(2,len(trivialmesh.zp)))

