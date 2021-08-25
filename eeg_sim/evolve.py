import numpy as np
from neurolib.models.wc import WCModel
from mne.simulation import SourceSimulator, simulate_stc
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.evolution import Evolution
from utils import (cnx_sample, build_band_samples, band_multivar_gauss_kl,
                   band_multivar_gauss_est)
import mne
import pickle
from os.path import isdir

def simulate_eeg(model, cnx, src, labels, fwd, info, scale_const,
                 noise_std, return_stc=False):

    noise_std *= scale_const
    model.params["Cmat"] = cnx
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

def simulate_sample(traj):
    model = evolution.getModelFromTraj(traj)
    params = model.params
    samp_cnx = cnx_sample(cnx_dict, samp_n)
    raws = []
    for i in range(samp_n):
        raw = simulate_eeg(model, samp_cnx[...,i], src, labels,
                           fwd, info, scale_const,
                           noise_std, return_stc=False)
        raws.append(raw)
    samples = build_band_samples(raws, bands, n_fft=500, log=True)
    distros = band_multivar_gauss_est(samples)
    kls = band_multivar_gauss_kl(distros, q_distro)
    fitness = np.array(list(kls.values())).mean()
    fitness = (fitness, )
    return fitness, model.outputs


if isdir("/home/jev"):
    root_dir = "/home/jev/hdd/"
elif isdir("/home/hannaj/"):
    root_dir = "/home/hannaj/hdd/"
elif isdir("/home/jeffhanna/"):
    root_dir = "/scratch/jeffhanna/"

mat_dir = "../mats/"
eeg_dir = root_dir + "memtacs/proc/"
subjects_dir = root_dir + "freesurfer/subjects/"

with open("{}mats.pickle".format(mat_dir), "rb") as f:
    cnx_dict = pickle.load(f)
# load raw template and source/forward info
raw = mne.io.Raw("{}MT-YG-102_0-raw.fif".format(eeg_dir))
info = raw.info
fwd = mne.read_forward_solution("{}fsaverage-fwd.fif".format(eeg_dir))
src = mne.read_source_spaces("{}fsaverage-src.fif".format(eeg_dir))
hcp_labels = mne.read_labels_from_annot("fsaverage", parc="HCP-MMP1",
                                    subjects_dir=subjects_dir)
with open("{}empirical_distro.pickle".format(mat_dir), "rb") as f:
    q_distro = pickle.load(f)


# fixed hyper-parameters
subsampling = 1000 / raw.info["sfreq"]
samp_n = 16
n_jobs = 32
noise_std = 0.2
scale_const = 1e-8

cnx = cnx_dict["cnx"]
cnx_d = cnx_dict["cnx_delay"]
reg_names = [r[0] for r in cnx_dict["Regions"]]
bands = {"theta":(4,8), "alpha":(8,13), "beta":(13,31)}

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

model = WCModel(Cmat=cnx, Dmat=cnx_d)
model.params["duration"]= 300 * 1000
model.params["dt"] = 0.5
model.params["sampling_dt"] = subsampling


param_names = ["K_gl", "c_excexc", "c_excinh", "c_inhexc", "c_inhinh",
               "exc_ext", "inh_ext"]
param_vals = [[0.05, 2], [4, 32], [4, 32], [4, 32], [0.05, 8], [0.05, 4],
              [0.05, 4]]

pars = ParameterSpace(param_names, param_vals)
evolution = Evolution(evalFunction=simulate_sample, parameterSpace=pars,
                      weightList=[-1.], model=model, POP_INIT_SIZE=64,
                      POP_SIZE=32, NGEN=20, ncores=n_jobs), filename="test.hdf")
evolution.run()
evolution.save()
