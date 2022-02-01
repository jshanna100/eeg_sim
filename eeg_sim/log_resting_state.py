import numpy as np
import mne
from os.path import isdir
import pickle
from os import listdir

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.
if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/reog/"

filenames = listdir(eeg_dir)
rests = []
for filename in filenames:
    if "-raw.fif" not in filename or "MT" not in filename:
        continue
    raw = mne.io.Raw(eeg_dir+filename)
    if raw.times[-1] < 500:
        rests.append(filename)

with open("{}resting_state_files.pickle".format(eeg_dir), "wb") as f:
    pickle.dump(rests, f)
