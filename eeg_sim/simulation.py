import numpy as np
from neurolib.models.wc import WCModel
import mne
from mne.simulation import SourceSimulator, simulate_stc
import pickle
import matplotlib.pyplot as plt
from os.path import isdir
plt.ion()

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"

mat_dir = base_dir + "eeg_sim/mats/"
eeg_dir = base_dir + "hdd/memtacs/proc/"
subjects_dir = base_dir+ "hdd/freesurfer/subjects/"


subsampling = 1
scale_const = 1e-8
noise_std = 0.2 * scale_const

with open("{}mats.pickle".format(mat_dir), "rb") as f:
    cnx_dict = pickle.load(f)

raw = mne.io.Raw("{}template-raw.fif".format(eeg_dir))
raw = raw.resample(1000/subsampling)
fwd = mne.read_forward_solution("{}fsaverage-fwd.fif".format(eeg_dir))
src = mne.read_source_spaces("{}fsaverage-src.fif".format(eeg_dir))


cnx = cnx_dict["cnx"]
cnx_d = cnx_dict["cnx_delay"]
reg_names = [r[0] for r in cnx_dict["Regions"]]

wc = WCModel(Cmat=cnx, Dmat=cnx_d)

wc.params['exc_ext'] = 0.65
wc.params['duration'] = 120 * 1000
wc.params['sigma_ou'] = 0.14
wc.params['K_gl'] = 3.15
wc.params["sampling_dt"] = subsampling

wc.run()
stc_data = wc.exc * scale_const

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


stc = simulate_stc(src, labels, stc_data, tmin=0, tstep=0.001)
stc.data += np.random.normal(0, noise_std, size=stc.data.shape)
raw_sim = mne.apply_forward_raw(fwd, stc, raw.info)
