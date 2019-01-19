import numpy as np

def polar2cart(rho,theta):
    """ Converts a polar rho,theta coordinate to a Cartesian x,y coordinate.

    Arguments rho and theta can be scalars/arrays/anything that numpy accepts.
    Normal broadcasting rules apply.

    Args:
        rho: radial coordinate
        theta: angular coordinate in radians

    Returns:
        tuple of (x, y)
    """
    return rho*np.cos(theta),rho*np.sin(theta)

def cart2polar(x,y):
    """ Converts a Cartesian x,y coordinate to a polar rho,theta coordinate.

    Arguments x and y can be scalars/arrays/anything that numpy accepts.
    Normal broadcasting rules apply.

    Args:
        x: x coordinate
        y: y coordinate

    Returns:
        tuple of (rho, theta)
    """
    return np.sqrt(x**2+y**2),np.arctan2(y,x)
