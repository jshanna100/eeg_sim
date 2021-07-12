import mne


subjects_dir = "/home/jev/freesurfer/subjects/"
eeg_dir = "/home/jev/hdd/Memtacs_EEG/"
n_jobs = 4


# source space
source_space = mne.setup_source_space("fsaverage", subjects_dir=subjects_dir,
                                      spacing="ico4", n_jobs=n_jobs)
mne.write_source_spaces("{}fsaverage-src.fif".format(eeg_dir), source_space)

# get EEG as template
raw = mne.io.read_raw_brainvision("{}MT-OG-232_TOM.vhdr".format(eeg_dir))
raw.drop_channels(['Re', 'Li', 'Vo', 'Vu'])
raw.set_montage("standard_1005")
raw.save("{}OG-232-raw.fif".format(eeg_dir))

# bem
surfs = mne.make_bem_model("fsaverage", subjects_dir=subjects_dir)
bem_sol = mne.make_bem_solution(surfs)
mne.write_bem_solution("{}fsaverage-bem.fif".format(eeg_dir), bem_sol)

# forward
fwd = mne.make_forward_solution(raw.info, "{}fsaverage-trans.fif".format(eeg_dir),
                                source_space, bem_sol, eeg=True, meg=False,
                                n_jobs=n_jobs)
mne.write_forward_solution("{}fsaverage-fwd.fif".format(eeg_dir), fwd)
