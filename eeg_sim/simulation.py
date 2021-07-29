import numpy as np
from neurolib.models.wc import WCModel
import mne
from mne.simulation import SourceSimulator, simulate_stc
import pickle
import matplotlib.pyplot as plt
from os.path import isdir
from utils import (cnx_sample, build_band_samples, plot_samples,
                   band_multivar_gauss_est, plot_covar_mats)
from joblib import Parallel, delayed
plt.ion()

def simulate_eeg(mod_params, cnx, src, labels, fwd, info, scale_const,
                 noise_std, return_stc=False):

    subsampling = 1000 / info["sfreq"]
    model = WCModel(Cmat=cnx, Dmat=mod_params["delay"])
    for mp_k, mp_v in mod_params.items():
        model.params[mp_k] = mp_v

    noise_std *= scale_const
    model.run(chunkwise=True, append_outputs=True)
    stc_data = model.exc * scale_const
    tstep = model.params["sampling_dt"] / 1000
    stc = simulate_stc(src, labels, stc_data, tmin=0, tstep=tstep)
    stc.data += np.random.normal(0, noise_std, size=stc.data.shape)
    raw_sim = mne.apply_forward_raw(fwd, stc, info)
    if return_stc:
        return stc, raw_sim
    else:
        return raw_sim

def simulate_sample(cnx_dict, samp_n, mod_params, src, labels, fwd, info,
                    scale_const=1e-8, noise_std=0.2, n_jobs=1, return_stc=False):
    samp_cnx = cnx_sample(cnx_dict, samp_n)
    if n_jobs == 1:
        raws = []
        for i in range(samp_n):
            raw = simulate_eeg(mod_params, samp_cnx[i], src, labels, fwd,
                               info, scale_const, noise_std,
                               return_stc=return_stc)
            raws.append(raw)
    else:
        raws = Parallel(n_jobs)(delayed(simulate_eeg)(mod_params, samp_cnx[i],
                                                      src, labels, fwd, info,
                                                      scale_const, noise_std)
                                                      for i in range(samp_n))
    return raws

if isdir("/home/jev"):
    root_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    root_dir = "/home/hannaj/"

mat_dir = root_dir + "eeg_sim/mats/"
eeg_dir = root_dir + "hdd/memtacs/proc/"
subjects_dir = root_dir + "hdd/freesurfer/subjects/"

with open("{}mats.pickle".format(mat_dir), "rb") as f:
    cnx_dict = pickle.load(f)
# load raw template and source/forward info
raw = mne.io.Raw("{}MT-YG-102_0-raw.fif".format(eeg_dir))
fwd = mne.read_forward_solution("{}fsaverage-fwd.fif".format(eeg_dir))
src = mne.read_source_spaces("{}fsaverage-src.fif".format(eeg_dir))

# fixed hyper-parameters
subsampling = 1000 / raw.info["sfreq"]
samp_n = 4
n_jobs = 4
log = True

cnx = cnx_dict["cnx"]
cnx_d = cnx_dict["cnx_delay"]
reg_names = [r[0] for r in cnx_dict["Regions"]]
bands = {"theta":(4,8), "alpha":(8,13), "beta":(13,31), "gamma":(30,100)}

mod_params = {}
mod_params["duration"]= 300 * 1000
mod_params["dt"] = 0.5
mod_params["sampling_dt"] = subsampling
mod_params["K_gl"] = 0.6
mod_params["delay"] = cnx_d
mod_params["exc_ext"] = 0.4

hcp_labels = mne.read_labels_from_annot("fsaverage", parc="HCP-MMP1",
                                    subjects_dir=subjects_dir)
labels = []
for reg in reg_names:
    hemi_str = "lh" if reg[0] == "L" else "rh"
    lab_str = "{}_ROI-{}".format(reg, hemi_str)
    this_label = None
    for lab in hcp_labels:
        if lab.name == lab_str:
            this_label = lab
            break
    if this_label is not None:
        labels.append(this_label)
    else:
        raise ValueError("Region name could not be located in labels.")
label_names = [label.name for label in labels]

raws = simulate_sample(cnx_dict, samp_n, mod_params, src, labels, fwd,
                       raw.info, scale_const=1e-8, noise_std=0.2,
                       return_stc=False, n_jobs=n_jobs)
samples = build_band_samples(raws, bands, n_jobs=n_jobs, n_fft=500, log=log)
distros = band_multivar_gauss_est(samples)
plot_covar_mats(distros)

with open("{}empirical_distro.pickle".format(mat_dir), "rb") as f:
    emp_distros = pickle.load(f)
