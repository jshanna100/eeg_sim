import numpy as np
from neurolib.models.wc import WCModel
import mne
from mne.simulation import SourceSimulator, simulate_raw
import pickle
import matplotlib.pyplot as plt
plt.ion()

mat_dir = "/home/jev/eeg_sim/mats/"
eeg_dir = "/home/jev/hdd/Memtacs_EEG/"
subjects_dir = "/home/jev/freesurfer/subjects/"

subsampling = 1

with open("{}mats.pickle".format(mat_dir), "rb") as f:
    cnx_dict = pickle.load(f)

raw = mne.io.Raw("{}OG-232-raw.fif".format(eeg_dir))
raw = raw.resample(1000/subsampling)
fwd = mne.read_forward_solution("{}fsaverage-fwd.fif".format(eeg_dir))
src = mne.read_source_spaces("{}fsaverage-src.fif".format(eeg_dir))


cnx = cnx_dict["cnx"]
cnx_d = cnx_dict["cnx_delay"]
reg_names = [r[0] for r in cnx_dict["Regions"]]

wc = WCModel(Cmat=cnx, Dmat=cnx_d)

wc.params['exc_ext'] = 0.65
wc.params['signalV'] = 0
wc.params['duration'] = 10 * 1000
wc.params['sigma_ou'] = 0.14
wc.params['K_gl'] = 3.15
wc.params["sampling_dt"] = subsampling

wc.run()
src_data = wc.exc * 1e-8

## check wc.t!! ####

labels = mne.read_labels_from_annot("fsaverage", parc="HCP-MMP1",
                                    subjects_dir=subjects_dir)
label_names = [label.name for label in labels]

ss = SourceSimulator(src, tstep=subsampling*1e-3)
for rn_idx, rn in enumerate(reg_names):
    hemi_str = "lh" if rn[0] == "L" else "rh"
    reg_str = "{}_ROI-{}".format(rn, hemi_str)
    for label in labels:
        if reg_str == label.name:
            this_label = label
            break
    ss.add_data(this_label, src_data[rn_idx,], np.array([[0,0,0]]))

raw_sim = simulate_raw(raw.info, ss, forward=fwd, n_jobs=8)
