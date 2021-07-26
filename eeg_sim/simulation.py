import numpy as np
from neurolib.models.wc import WCModel
import mne
from mne.simulation import SourceSimulator, simulate_stc
import pickle
import matplotlib.pyplot as plt
from os.path import isdir
from utils import cnx_sample
plt.ion()

def simulate_eeg(model, src, labels, fwd, info, scale_const, noise_std):
    noise_std *= scale_const
    model.run(chunkwise=True, append_outputs=True)
    stc_data = model.exc * scale_const
    tstep = model.params["sampling_dt"] / 1000
    stc = simulate_stc(src, labels, stc_data, tmin=0, tstep=tstep)
    stc.data += np.random.normal(0, noise_std, size=stc.data.shape)
    raw_sim = mne.apply_forward_raw(fwd, stc, info)
    return stc, raw_sim

def simulate_sample(cnx_dict, samp_n, model, src, labels, fwd, info,
                    scale_const=1e-8, noise_std=0.2):
    samp_cnx = cnx_sample(cnx_dict, samp_n)
    raws = []
    for sim_idx in range(samp_cnx.shape[-1]):
        model.params["Cmat"] = samp_cnx[...,sim_idx].copy()
        _, raw = simulate_eeg(model, src, labels, fwd, info, scale_const,
                              noise_std)
        raws.append(raw)
    return raws

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"

mat_dir = base_dir + "eeg_sim/mats/"
eeg_dir = base_dir + "hdd/memtacs/proc/"
subjects_dir = base_dir+ "hdd/freesurfer/subjects/"

with open("{}mats.pickle".format(mat_dir), "rb") as f:
    cnx_dict = pickle.load(f)
# load raw template and source/forward info
raw = mne.io.Raw("{}MT-YG-102_0-raw.fif".format(eeg_dir))
fwd = mne.read_forward_solution("{}fsaverage-fwd.fif".format(eeg_dir))
src = mne.read_source_spaces("{}fsaverage-src.fif".format(eeg_dir))

# fixed hyper-parameters
subsampling = 1000 / raw.info["sfreq"]
samp_n = 2
model = WCModel(Cmat=cnx_dict["cnx"], Dmat=cnx_dict["cnx_delay"])
model.params.duration= 300 * 1000
model.params.dt = 0.5
model.params.sampling_dt = subsampling
model.params.K_gl = 0 # global coupling

cnx = cnx_dict["cnx"]
cnx_d = cnx_dict["cnx_delay"]
reg_names = [r[0] for r in cnx_dict["Regions"]]


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

raws = simulate_sample(cnx_dict, samp_n, model, src, labels, fwd, raw.info,
                       scale_const=1e-8, noise_std=0.2)
