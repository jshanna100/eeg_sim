import numpy as np
from sklearn.decomposition import PCA
import mne
from mne.simulation.raw import _SimForwards, _check_head_pos
from os.path import isdir
from os import listdir
import matplotlib.pyplot as plt
from utils import *
plt.ion()

class SaccadeGenerator():
    def __init__(self, dipoles, rates, use_comps, sph_pts, sfreq=1000):
        '''
        raw: MNE Python Raw instance
             Simulated Saccades will be added on to this.
        dipoles: numpy array Ex2x3xN
                 where E is epochs and N is samples.
                 These represent the dipole vectors of measured saccades.
        use_comps: int
                   the first x components to consider signal, the rest will
                   be considered noise components and not used to generate
        '''
        self.rates = rates
        self.sfreq = sfreq
        self.sph_pts = sph_pts
        self.use_comps = use_comps

        # first do PCA on the dipole vectors
        pca = PCA()
        E, N, L = dipoles.shape[0], dipoles.shape[-1], dipoles.shape[1]
        # put into sample * feature shape for PCA
        dip_flat = np.transpose(dipoles, (1,2,0,3))
        dip_flat = dip_flat.reshape(dip_flat.shape[0] * dip_flat.shape[1],
                                    dip_flat.shape[2] * dip_flat.shape[3])

        print("Fitting PCA and transforming data.")
        trans = pca.fit_transform(dip_flat.T)

        expl_var = np.sum(pca.explained_variance_ratio_[:use_comps])
        print("Using {} components, {:.3f} of variance.".format(use_comps,
                                                                expl_var))

        # reconstruct with noise components only to find the magnitude of how much
        # random noise we should add
        print("Calculating noise variance.")
        noise_trans = trans.copy()
        noise_trans[:, :self.use_comps] = 0
        noise = pca.inverse_transform(noise_trans) * 1e-12
        noise_std = noise.std(axis=0)

        self.noise_std = noise_std
        # put the signal transformations into E*6*N shape for later use
        trans[:, self.use_comps:] = 0 # zero out noise
        trans = trans.reshape(E, N, -1)
        trans = np.transpose(trans, (0, 2, 1))
        self.trans = trans
        self.pca = pca

    def get_saccade(self):
        sacc_trans = np.zeros(self.trans.shape[1:])
        # grab components from random trials
        for idx in range(self.use_comps):
            rand_int = np.random.randint(len(self.trans))
            sacc_trans[idx,] = self.trans[rand_int, idx,]
        # transform to saccade dipoles
        # multiply by 1.2 to make up for unexplained shortfall
        dip_array = self.pca.inverse_transform(sacc_trans.T) * 1e-12 * 1.2
        return dip_array

    def generate(self, raw, annotate=False):
        info = raw.info
        if info["sfreq"] != self.sfreq:
            raise ValueError("Sampling rate of raw and dipoles do not match.")
        # set up the dipole time courses and overlay appropriate noise
        N = len(raw)
        dip = np.zeros((self.trans.shape[1], N))
        for idx, noise_std in enumerate(self.noise_std):
            dip[idx, ] = np.random.normal(0, noise_std, size=N)

        # sample inhomogenous poisson process for saccade occurrences
        rates = self.rates
        rate = np.random.normal(np.mean(rates), np.std(rates))
        sacc_occs = np.random.uniform(size=len(raw)) < rate / info["sfreq"]
        # don't want saccade in first or last second
        sacc_occs[:int(info["sfreq"])] = False
        sacc_occs[-int(info["sfreq"]):] = False
        so_inds = np.where(sacc_occs)[0]

        # cycle through all saccade occurences and simulate a saccade
        print("Generating saccades.")
        for idx in so_inds:
            sacc_dip = self.get_saccade().T
            dip[:, idx:idx+sacc_dip.shape[1]] = sacc_dip

        print("Calculating forward model.")
        ## use dip sources to create EEG
        # fit sphere
        R, r0, _ = mne.bem.fit_sphere_to_headshape(raw.info)
        # make sphere
        sphere = mne.bem.make_sphere_model(r0, head_radius=R,
                                           relative_radii=(0.97, 0.98, 0.99, 1.),
                                           sigmas=(0.33, 1.0, 0.004, 0.33),
                                           verbose=False)

        # eyeball dipole locations
        exg_rr = self.sph_pts
        # translate to this subject's space
        exg_rr /= np.sqrt(np.sum(exg_rr**2, axis=1, keepdims=True))
        exg_rr *= 0.96 * R
        exg_rr += r0

        # random orientations as placeholders
        exg_nn = np.random.randn(*exg_rr.shape)
        exg_nn = (exg_nn.T / np.linalg.norm(exg_nn, axis=1)).T

        src = mne.setup_volume_source_space(pos={"rr":exg_rr, "nn":exg_nn},
                                            sphere_units="mm")
        dev_head_ts, offsets = _check_head_pos(None, info, 0, raw.times)
        get_fwd = _SimForwards(dev_head_ts, offsets, info, None, src, sphere, 0.005,
                               8, mne.pick_types(info, eeg=True))
        fwd = next(get_fwd.iter)

        # use forward model to translate dipoles into eeg
        print("Simulating raw.")
        raw_array = np.dot(fwd["sol"]["data"], dip)
        new_info = raw.copy().pick_types(eeg=True).info
        sacc_raw = mne.io.RawArray(raw_array, new_info)
        if annotate:
            times = raw.times[so_inds]
            for time in times:
                sacc_raw.annotations.append(time, raw.times[sacc_dip.shape[1]],
                                            "MicroSaccade")
        return sacc_raw

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
mat_dir = base_dir+"eeg_sim/mats/"
eeg_dir = base_dir+"hdd/memtacs/proc/reog/"

rates = np.load("{}rates.npy".format(mat_dir))
dipoles = np.load("{}dipole_vecs.npy".format(mat_dir))
sph_pts = np.load("{}sph_points.npy".format(mat_dir))

# file_names = listdir(eeg_dir)
# for file_name in file_names:
#     if not re.search("\d-epo.fif", file_name):
#         continue
#     epo = mne.read_epochs(eeg_dir + file_name)

raw = mne.io.Raw(eeg_dir+"MT-YG-102_2-raw.fif", preload=True)
#add_null_reference_chan(raw, "Nose_ref")
raw.set_eeg_reference()
raw.apply_proj()

sacc_gen = SaccadeGenerator(dipoles, rates, 3, sph_pts)
sacc_raw = sacc_gen.generate(raw, annotate=True)

epo = mne.read_epochs(eeg_dir+"102_2-epo.fif")
epo.set_eeg_reference()
epo.apply_proj()
evo = epo.average()
evo.plot_joint()

events, _ = mne.events_from_annotations(sacc_raw)
epo_sim = mne.Epochs(sacc_raw, events, tmin=-0.25, tmax=0.25, baseline=None)
evo_sim = epo_sim.average()
evo_sim.plot_joint()
