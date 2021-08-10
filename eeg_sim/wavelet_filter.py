from scipy.signal import cwt
import numpy as np
import mne

def sacc_wavelet(point_n, sigma):
    x = np.linspace(-3, 3, point_n)
    y = x / (sigma**3 * np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2 * sigma**2)
    return y
