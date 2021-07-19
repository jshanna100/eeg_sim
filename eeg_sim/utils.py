import mne
from mne.time_frequency import psd_welch
import numpy as np

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
