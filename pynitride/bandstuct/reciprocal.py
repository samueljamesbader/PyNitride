r""" Utilities for describing the reciprocal space structure of a crystal"""
import numpy as np

def generate_path(points,n):
    r""" Generates a 2D

    :param points:
    :param n:
    :return:
    """
    points=[np.array(p) for p in points]
    return np.vstack(
            [np.array([np.linspace(0,1,n-1,endpoint=False)]).T*(pf-ps)+ps
        for ps,pf in zip(points[:-1],points[1:])]+[points[-1]])

#def get_symmetry_point(material,label):
    #asser

