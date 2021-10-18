from os.path import isdir
import numpy as np
import mne

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.
if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/reog/"
epo = mne.read_epochs("{}grand_saccade-epo.fif".format(eeg_dir))
epo.set_eeg_reference()
evo = epo.average()

# fit sphere
R, r0, _ = mne.bem.fit_sphere_to_headshape(epo.info)
# make sphere
sphere = mne.make_sphere_model(r0=r0, head_radius=R,
                               relative_radii=(0.97, 0.98, 0.99, 1.))
# eyeball dipole locations
exg_rr = np.array([[np.cos(np.pi / 3.), np.sin(np.pi / 3.), 0.],
                   [-np.cos(np.pi / 3.), np.sin(np.pi / 3), 0.]])
exg_rr /= np.sqrt(np.sum(exg_rr * exg_rr, axis=1, keepdims=True))
exg_rr *= 0.96 * R
exg_rr += r0

# covariance
cov = mne.compute_covariance(epo)

# dipoles
dip_0, resid_0 = mne.fit_dipole(evo, cov, sphere, pos=exg_rr[0,], min_dist=0)
dip_1, resid_1 = mne.fit_dipole(evo, cov, sphere, pos=exg_rr[1,], min_dist=0)
