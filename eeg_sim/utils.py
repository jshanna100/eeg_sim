import mne
from mne.time_frequency import psd_welch
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mayavi.mlab import points3d, plot3d, mesh, quiver3d, figure

def sigma2freq(sigma_min, sigma_max, samp_len, point_n, resolution=250):
    freq_table = {}
    sig_range = np.linspace(sigma_min, sigma_max, resolution)
    for sig in sig_range:
        y = sacc_wavelet(point_n, sig)
        y /= y.max()
        active = (y!=0).sum()
        freq = (point_n / active) * samp_len
        freq_table[sig] = freq
    return freq_table

def sacc_wavelet(point_n, sigma):
    x = np.linspace(-0.5, 0.5, point_n)
    y = x / (sigma**3 * np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2 * sigma**2)
    return y

def multivar_gauss_kl(p, q):
    mu_p, sigma_p, mu_q, sigma_q = p["mu"], p["sigma"], q["mu"], q["sigma"]
    dim_n = len(mu_p)
    term_0 = np.log(abs(np.linalg.det(sigma_q)) / abs(np.linalg.det(sigma_p)))
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
    resid = samp_vecs - np.tile(np.expand_dims(mu, 0).T,
                                (1, samp_vecs.shape[1]))
    sigma = np.dot(resid, resid.T) / samp_vecs.shape[1]
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
    # for a list of raw files and a dictonary of frequency bands,
    # calculate the band power for each channel
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
            ax.hist(output[band][ch], density=True, alpha=0.5)
            ax.plot(x, y)
            ax.set_xlim(xlim)
        plt.tight_layout()

def draw_pair(points, size, color, fig):
    points3d(points[:,0], points[:,1], points[:,2], scale_factor=size,
             color=color, figure=fig)
    plot3d(points[:,0], points[:,1], points[:,2],
           tube_radius=None, color=color, figure=fig)

def draw_eeg(info, size, color, fig):
    hsp = mne.bem.get_fitting_dig(info)
    for d_idx in range(len(hsp)):
        points3d(hsp[d_idx, 0], hsp[d_idx, 1], hsp[d_idx, 2], scale_factor=size,
                 color=color)

def draw_sphere(R, r0, color, alpha, fig):
    [phi, theta] = np.mgrid[0:2 * np.pi:12j, 0:np.pi:12j]
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return mesh(R * x + r0[0], R * y + r0[1], R * z + r0[2], color=color,
                opacity=alpha, figure=fig)

def get_rand_rrs(size, R, r0, z_excl=0.):
    r_rr = np.zeros(size)
    for r_idx in range(size[0]):
        r_r = np.random.normal(size=(1, size[1]))
        r_r /= np.sqrt(np.sum(r_r**2, axis=1, keepdims=True))
        r_r *= 0.96 * R
        r_r += r0
        while r_r[0, -1] < z_excl:
            r_r = np.random.normal(size=(1, size[1]))
            r_r /= np.sqrt(np.sum(r_r**2, axis=1, keepdims=True))
            r_r *= 0.96 * R
            r_r += r0
        r_rr[r_idx,] = r_r
    return r_rr
