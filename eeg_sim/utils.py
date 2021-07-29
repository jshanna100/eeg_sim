import mne
from mne.time_frequency import psd_welch
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def multivar_gauss_kl(p, q):
    mu_p, sigma_p, mu_q, sigma_q = p["mu"], p["sigma"], q["mu"], q["sigma"]
    dim_n = len(mu_p)
    term_0 = np.log(np.linalg.det(sigma_q) / np.linalg.det(sigma_p))
    diff = mu_p - mu_q
    term_1 = np.dot(diff, np.linalg.inv(sigma_q)).dot(diff)
    term_2 = np.trace(np.matmul(np.linalg.inv(sigma_q), sigma_p))
    kl = 0.5 * (term_0 - dim_n + term_1 + term_2)
    return kl

def band_multivar_gauss_kl(p_distros, q_distros):
    band_kls = {}
    for band_k in p_distros.keys():
        if band_k != "ch_names":
            band_kls[band_k] = multivar_gauss_kl(p_distros[band_k],
                                                 q_distros[band_k])
    return band_kls

def multivar_gauss_est(samples):
    samp_vecs = np.stack(list(samples.values()))
    mu = samp_vecs.mean(axis=1)
    sigma = np.dot(samp_vecs, samp_vecs.T) / samp_vecs.shape[1]
    return {"mu":mu, "sigma":sigma}

def band_multivar_gauss_est(band_samples):
    band_distros = {"ch_names":band_samples["ch_names"]}
    for band_k, band_v in band_samples.items():
        if band_k != "ch_names":
            band_distros[band_k] = multivar_gauss_est(band_v)
    return band_distros

def calc_bandpower(inst, picks, bands, n_fft=500, n_jobs=1, log=False):
    if isinstance(inst, mne.io.BaseRaw):
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

def build_band_samples(raws, bands, n_fft=500, log=False, n_jobs=1):
    output = {k:{} for k in bands.keys()}
    for raw in raws:
        ch_names = [raw.ch_names[idx] for idx in mne.pick_types(raw.info,
                                                                eeg=True)]
        print(len(ch_names))
        bandpower = calc_bandpower(raw, ch_names, bands, n_jobs=n_jobs, log=log)
        for band in bands.keys():
            for ch in ch_names:
                if ch not in output[band]:
                    output[band][ch] = []
                ch_idx = bandpower["chan_names"].index(ch)
                output[band][ch].extend(bandpower[band][:, ch_idx])
    output["ch_names"] = ch_names
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

def plot_covar_mats(band_distros, rows=2, cols=2, figsize=(19.2,19.2)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = [ax for axe in axes for ax in axe]
    bands = [band for band in band_distros.keys() if band != "ch_names"]
    for band, ax in zip(bands, axes):
        ax.imshow(band_distros[band]["sigma"])
        ax.set_title(band)
        ax.set_xticks(np.arange(len(band_distros["ch_names"])))
        ax.set_xticklabels(band_distros["ch_names"], rotation=90)
        ax.set_yticks(np.arange(len(band_distros["ch_names"])))
        ax.set_yticklabels(band_distros["ch_names"])

def plot_samples(output, bands, ch_names, xlim=(-3.5, 3.5)):
    for band in bands.keys():
        fig, axes = plt.subplots(4,7, figsize=(19.2, 10.8))
        axes = [a for ax in axes for a in ax]
        plt.suptitle(band)
        for ax, ch in zip(axes, ch_names):
            output[band][ch] = np.array(output[band][ch])
            mu = output[band][ch].mean()
            std = output[band][ch].std()
            ax.set_title(ch)
            this_norm = norm(loc=mu,
                             scale=std)
            x = np.linspace(this_norm.ppf(0.001), this_norm.ppf(0.999), 500)
            y = this_norm.pdf(x)
            ax.hist(output[band][ch], density=True, alpha=0.5, bins=30)
            ax.plot(x, y)
            ax.set_xlim(xlim)
        plt.tight_layout()
