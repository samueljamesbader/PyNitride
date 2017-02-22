import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss_fit_fwhm(x, y, fitplot=False):
    y = y - np.min(y)
    i = np.argmax(y)

    il = np.argmax(y[:i] > y[i] / 2)
    ir = np.argmin(y[i + 1:] > y[i] / 2) + i + 1

    a0 = y[i]
    xroi = x[il:ir + 1]
    yroi = y[il:ir + 1]
    mu0 = np.sum(xroi * yroi) / np.sum(yroi)
    sigma0 = np.sqrt(np.sum((xroi - mu0) ** 2 * yroi) / np.sum(yroi))

    def gauss(x, a, mu, sigma):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    popt, pcov = curve_fit(gauss, xroi, yroi, p0=[a0, mu0, sigma0])

    if fitplot:
        plt.figure()
        plt.plot(x, y, label='supplied data')
        plt.plot(xroi, yroi, 'o', label='roi')
        plt.plot(xroi, gauss(xroi, a0, mu0, sigma0), '--', label='initial guess')
        plt.plot(x, gauss(x, *popt), linewidth=2, label='fit')
        plt.legend(loc='best')

    return {'a': popt[0], 'mu': popt[1], 'sigma': popt[2],
            'a_unc': np.sqrt(pcov[0,0]),'mu_unc': np.sqrt(pcov[1,1]),'sigma_unc': np.sqrt(pcov[2,2]),
            'vars': ['a','mu','sigma'],
            'popt': popt, 'pcov': pcov, 'func': lambda x: gauss(x, *popt)}

def roi(x,y,xmin,xmax):
    mask=(x>xmin)*(x<xmax)
    return x[mask],y[mask]
