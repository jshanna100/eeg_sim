import mne
from mne.time_frequency import psd_welch
import numpy as np
from scipy.stats import norm

def calc_bandpower(inst, picks, bands, n_fft=500, n_jobs=1, log=False):
    if isinstance(inst, mne.io.Raw):
        epo = mne.make_fixed_length_epochs(inst, duration=5)
    else:
        epo = inst
    output = {"chan_names":picks}
    min_freq = np.array([x[0] for x in bands.values()]).min()
    max_freq = np.array([x[1] for x in bands.values()]).max()
    psd, freqs = psd_welch(epo, picks=picks, n_fft=n_fft,
                           fmin=min_freq, fmax=max_freq,
                           n_jobs=n_jobs)
    psd *= 1e+12
    if log:
        psd = np.log(psd)
    for band_k, band_v in bands.items():
        fmin, fmax = band_v
        inds = np.where((freqs>=fmin) & (freqs<=fmax))[0]
        output[band_k] = psd[...,inds].mean(axis=-1)
    return output

def cnx_sample(cnx_dict, samp_size):
    tri = np.triu(cnx_dict["cnx"])
    tri_inds = np.where(tri)
    samp_mat = np.zeros((*tri.shape, samp_size))
    for x,y in zip(*tri_inds):
        this_norm = norm(tri[x,y], cnx_dict["cnx_var"][x,y]*tri[x,y])
        samp_mat[x,y,] = this_norm.rvs(size=samp_size)
    samp_mat += np.transpose(samp_mat, [1, 0, 2])
    return samp_mat
