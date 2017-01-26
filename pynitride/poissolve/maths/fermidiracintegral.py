import numpy as np
import warnings

def fd12(x):
    # Wong et al Solid-State Electronics Vol. 37, No. I, pp. 61~64, 1994
    a = np.array([1, .353568, .192439, .122973, .077134, .036228, .008346])
    b1 = np.array([.76514793, .60488667, .19003355, 2.00193968e-2, -4.12643816e-3, -4.70958992e-4, 1.50071469e-4])
    b2 = np.array([.78095732, .57254453, .21419339, 1.38432741e-2, -5.54949386e-3, 6.48814900e-4, -2.84050520e-5])
    c = np.array([.752253, .928195, .680839, 25.7829, -553.636, 3531.43, -3254.65])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.choose((0 + (x > 0) + (x > 2) + (x > 5)), [
            np.sum(list(map(lambda i: (-1) ** i * a[i] * np.exp((i + 1) * x), range(7))), 0),
            np.polyval(b1[::-1], x),
            np.polyval(b2[::-1], x),
            np.sum(list(map(lambda i: c[i] * abs(x + (x == 0)) ** (1.5 - 2 * i), range(7))), 0)
        ])


def fd12p(x):
    # Wong et al Solid-State Electronics Vol. 37, No. I, pp. 61~64, 1994
    a = np.array([1, .353568, .192439, .122973, .077134, .036228, .008346])
    b1 = np.array([.76514793, .60488667, .19003355, 2.00193968e-2, -4.12643816e-3, -4.70958992e-4, 1.50071469e-4])
    b2 = np.array([.78095732, .57254453, .21419339, 1.38432741e-2, -5.54949386e-3, 6.48814900e-4, -2.84050520e-5])
    c = np.array([.752253, .928195, .680839, 25.7829, -553.636, 3531.43, -3254.65])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.choose((0 + (x > 0) + (x > 2) + (x > 5)), [
            np.sum(list(map(lambda i: (-1) ** i * (i + 1) * a[i] * np.exp((i + 1) * x), range(7))), 0),
            np.polyval(np.polyder(b1[::-1]), x),
            np.polyval(np.polyder(b2[::-1]), x),
            np.sum(list(map(lambda i: c[i] * (1.5 - 2 * i) * abs(x + (x == 0)) ** (1.5 - 2 * i - 1), range(7))), 0)
        ])