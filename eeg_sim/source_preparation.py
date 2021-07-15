import mne
from os.path import isdir

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"

subjects_dir = base_dir + "hdd/freesurfer/subjects/"
eeg_dir = base_dir + "hdd/memtacs/proc/"

n_jobs = 24


# source space
source_space = mne.setup_source_space("fsaverage", subjects_dir=subjects_dir,
                                      spacing="ico4", n_jobs=n_jobs)
mne.write_source_spaces("{}fsaverage-src.fif".format(eeg_dir), source_space)

# get EEG as template
raw = mne.io.Raw("{}template-raw.fif".format(eeg_dir))

# bem
surfs = mne.make_bem_model("fsaverage", subjects_dir=subjects_dir)
bem_sol = mne.make_bem_solution(surfs)
mne.write_bem_solution("{}fsaverage-bem.fif".format(eeg_dir), bem_sol)

# forward
fwd = mne.make_forward_solution(raw.info, "{}fsaverage-trans.fif".format(eeg_dir),
                                source_space, bem_sol, eeg=True, meg=False,
                                n_jobs=n_jobs)
mne.write_forward_solution("{}fsaverage-fwd.fif".format(eeg_dir), fwd)
