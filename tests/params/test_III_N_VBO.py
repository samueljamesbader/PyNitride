import numpy as np
from scipy.stats import linregress
from pynitride.core.mesh import MaterialBlock, Mesh, UniformLayer
from pynitride.physics.material import AlInGaN
from pynitride import nm, eV
from pynitride.physics.solvers import Linear_Fermi, PoissonSolver
from pynitride.physics.strain import Pseudomorphic
from pynitride.physics.thermal import ConstantT

def get_band_offsets(thin_kwargs,bulk_kwargs):
    m=Mesh([MaterialBlock("heterojunction",AlInGaN(),[
        UniformLayer("thin",1*nm,**thin_kwargs),
        UniformLayer("bulk",1*nm,**bulk_kwargs),
        ])],max_dz=.1*nm,boundary=[.7,"thick"])
    Pseudomorphic(m).solve()
    ConstantT(m).solve()
    Linear_Fermi(m).solve()
    PoissonSolver(m).solve()

    m1,m2=m.submesh_cover([1*nm],names=['mat1','mat2'])
    Ec1=linregress(m1.zm-1*nm,m1.Ec).intercept # type: ignore
    Ec2=linregress(m2.zm-1*nm,m2.Ec).intercept # type: ignore
    CBO=Ec1-Ec2
    Ev1=linregress(m1.zm-1*nm,m1.Ev).intercept # type: ignore
    Ev2=linregress(m2.zm-1*nm,m2.Ev).intercept # type: ignore
    VBO=Ev1-Ev2
    return CBO,VBO

def test_AlN_on_GaN_BOs():

# For VBOs, see pg 425 of
# Morkoc, "Handbook of Nitride Semiconductors and Devices,
#   Vol II: Electronic and Optical Processes in Nitrides"
    CBO,VBO = get_band_offsets(thin_kwargs={"x":1,"y":0},bulk_kwargs={"x":0,"y":0})
    assert np.isclose(CBO,2.01*eV,atol=0.01*eV)
    assert np.isclose(VBO, -.18*eV,atol=0.01*eV) # Morkoc says -.2eV
def test_GaN_on_AlN_BOs():
    CBO,VBO = get_band_offsets(thin_kwargs={"x":0,"y":0},bulk_kwargs={"x":1,"y":0})
    assert np.isclose(CBO,-1.70*eV,atol=0.01*eV)
    assert np.isclose(VBO,  .85*eV,atol=0.01*eV)
def test_InN_on_GaN_BOs():
    CBO,VBO = get_band_offsets(thin_kwargs={"x":0,"y":1},bulk_kwargs={"x":0,"y":0})
    assert np.isclose(VBO,  .5*eV,atol=0.01*eV)

if __name__=="__main__":
    test_AlN_on_GaN_BOs()
    test_GaN_on_AlN_BOs()
    test_InN_on_GaN_BOs()