r""" Utilities for describing the reciprocal space structure of a crystal"""
import numpy as np

def generate_path(points,n):
    r""" Generates a 2D

    :param points: a sequence of fixed-points in k-space through which to interpolate
    :param n: the number of points to place along each subpath in k-space (including both endpoints).  So the total
        number of points will be :math:`(p-1)*(n-1)+1` where :math:`p` is ``len(points)``.
    :return: a 2D Numpy array where each row is a k-vector.
    """
    points=[np.array(p) for p in points]
    return np.vstack(
            [np.array([np.linspace(0,1,n-1,endpoint=False)]).T*(pf-ps)+ps
        for ps,pf in zip(points[:-1],points[1:])]+[points[-1]])

def get_symmetry_point(label,material):
    vf=material._pmdb['crystal='+material['crystal'],'brillouin','symmetry points',label]
    return vf(**{k:material['lattice',k] for k in vf.__code__.co_varnames})
