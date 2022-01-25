from os.path import isdir
from os import listdir
import numpy as np
import mne
import re
from mne.simulation.raw import _SimForwards, _check_head_pos
from mne.minimum_norm.inverse import (_check_or_prepare, _assemble_kernel,
                                      _pick_channels_inverse_operator)
from mne.inverse_sparse import mixed_norm
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_epochs, rap_music
import matplotlib.pyplot as plt
import pickle
plt.ion()
from mayavi import mlab
from mayavi.mlab import points3d, plot3d, mesh, quiver3d, figure
from utils import *
from itertools import product
from sklearn.decomposition import PCA

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/reog/"
mat_dir = base_dir+"eeg_sim/mats/"
img_dir = eeg_dir+"anim/"

calc = True

loc_time = (-0.01, 0.01)
#loc_time = (0.01, 0.05)
#loc_time = (-0.01, 0.05)

method = "MNE"
lambda2_evo = 1.0 / 3.0 ** 2
lambda2_epo = 1.0 / 3.0 ** 2

if calc:
    # # random points on a unit sphere
    # pts_n = 3000
    # sph_pts = np.random.normal(0, 1, size=(pts_n, 3))
    # sph_pts = sph_pts / np.linalg.norm(sph_pts, axis=1)[:, np.newaxis]
    # # eliminate points where we're not interested in searching
    # sph_pts = sph_pts[sph_pts[:,2]>-.8]
    # sph_pts = sph_pts[sph_pts[:,2]<-.2]
    # sph_pts = sph_pts[sph_pts[:,1]>.6]
    # exg_rr = sph_pts

    sd = 2.8
    Z = -.5
    og_exg_rr = np.array([[np.cos(np.pi / sd), np.sin(np.pi / sd), Z],
                       [-np.cos(np.pi / sd), np.sin(np.pi / sd), Z]])
    og_exg_rr /= np.sqrt(np.sum(og_exg_rr**2, axis=1, keepdims=True))

    file_names = listdir(eeg_dir)
    expls = []
    ests = []
    vecs = []
    for file_name in file_names:
        if not re.search("\d-epo.fif", file_name):
            continue
        epo = mne.read_epochs(eeg_dir + file_name)
        #add_null_reference_chan(epo, "Nose_ref")
        epo.set_eeg_reference(projection=True)
        cov = mne.compute_covariance(epo.copy().crop(tmin=-0.2, tmax=-0.15))
        evo = epo.average()

        evo.crop(tmin=loc_time[0], tmax=loc_time[1])
        epo.crop(tmin=loc_time[0], tmax=loc_time[1])
        info, times, first_samp = evo.info, evo.times, 0

        # fit sphere
        R, r0, _ = mne.bem.fit_sphere_to_headshape(evo.info, dig_kinds="eeg")
        # make sphere
        sphere = mne.bem.make_sphere_model(r0, head_radius=R,
                                           relative_radii=(0.97, 0.98, 0.99, 1.),
                                           sigmas=(0.33, 1.0, 0.004, 0.33),
                                           verbose=False)


        # translate to our sphere
        exg_rr = r0[np.newaxis, :] + og_exg_rr * (R * 0.96) # originally 0.96
        # random orientations; will be allowed to vary with localisation anyway
        exg_nn = np.random.randn(*exg_rr.shape)
        exg_nn = (exg_nn.T / np.linalg.norm(exg_nn, axis=1)).T

        # # viz all points
        # fig = figure("all dipoles")
        # draw_sphere(R, r0, (0,0,1), 0.1, fig)
        # draw_eeg(evo.info, 0.01, (0,0,0), fig)
        # for idx in range(len(exg_rr)):
        #     draw_point(exg_rr[idx]*0.95, 0.005, (1,0,0), fig)

        ## localise epochs
        # random orientations; will be allowed to vary with localisation anyway
        exg_nn = np.random.randn(*exg_rr.shape)
        exg_nn = (exg_nn.T / np.linalg.norm(exg_nn, axis=1)).T

        src = mne.setup_volume_source_space(pos={"rr":exg_rr, "nn":exg_nn},
                                            sphere_units="mm")
        dev_head_ts, offsets = _check_head_pos(None, info, first_samp, times)
        get_fwd = _SimForwards(dev_head_ts, offsets, info, None, src, sphere,
                               0.005, 8, mne.pick_types(info, eeg=True))
        fwd = next(get_fwd.iter)

        if method == "beamformer":
            cov_data = mne.compute_covariance(epo.copy())
            filts = make_lcmv(epo.info, fwd, cov_data, noise_cov=cov,
                              pick_ori="vector", reduce_rank=True)
            stcs = apply_lcmv_epochs(epo, filts)
            inds = np.arange(len(epo))
            vecs.append(np.array([(stcs[idx]).data.astype(np.float32)
                                  for idx in inds]))
        elif method in ["MNE", "dSPM", "sLORETA", "eLORETA"]:
            inv = mne.minimum_norm.make_inverse_operator(epo.info, fwd, cov,
                                                         fixed=False, depth=0.8,
                                                         use_cps=False)
            stcs = mne.minimum_norm.apply_inverse_epochs(epo, inv, lambda2_epo,
                                                         method="MNE",
                                                         pick_ori="vector")
            expls = np.array([x[1] for x in stcs])
            inds = np.where(expls>25.)[0] # only epos that explain at least x var

            if len(inds):
                vecs.append(np.array([(stcs[idx][0].data*1e+12).astype(np.float32)
                                      for idx in inds]))
        elif method == "sparse":
            stc, resid = mixed_norm(evo, fwd, cov, pick_ori="vector",
                                    return_residual=True)


        # # find strongest amplitude
        # normed = np.linalg.norm(stc.data, axis=1)
        # max_idx = np.argmax(normed.mean(axis=0), axis=0)
        # # get inds for strongest n points
        # big_inds = np.argsort(normed[:, max_idx])[-100:]
        # # scaling for alpha
        # min, max = normed[big_inds[0], max_idx], normed[big_inds[-1], max_idx]
        # # # draw dipoles
        # fig = figure(file_name)
        # draw_eeg(evo.info, 0.01, (0,0,0), fig)
        # draw_sphere(R, r0, (0,0,1), 0.1, fig)
        # for idx in big_inds:
        #     alpha = (normed[idx, max_idx] - min) / (max - min)
        #     draw_point(exg_rr[idx,], 0.01, (1,0,0), fig, alpha=alpha)

    vecs = np.vstack(vecs)
    np.save("{}dipole_vecs.npy".format(mat_dir), vecs)
    np.save("{}sph_points.npy".format(mat_dir), og_exg_rr)
else:
    vecs = np.load("{}dipole_vecs.npy".format(mat_dir))
    sph_pts = np.load("{}sph_points.npy".format(mat_dir))

    vecs_shape = vecs.shape
    vecs = np.transpose(vecs, (1,2,0,3))
    vecs_shape = vecs.shape
    vecs = vecs.reshape(vecs.shape[0]*vecs.shape[1], vecs.shape[2]*vecs.shape[3])
    pca = PCA()
    trans = pca.fit_transform(vecs.T)

    comps = pca.components_.reshape(-1, *vecs_shape[:2])

    # # draw component weights
    for comp_idx in range(5):
        fig = figure("Component {}".format(comp_idx))
        #draw_eeg(evo.info, 0.01, (0,0,0), fig)
        draw_sphere(1, (0,0,0), (0,0,1), 0.1, fig)
        this_comp = comps[comp_idx,]
        # this_comp = np.linalg.norm(this_comp, axis=1)
        max, min = this_comp.max(), this_comp.min()
        inds = np.argsort(np.linalg.norm(this_comp, axis=1))[-100:]
        for idx in inds:
            alpha = (this_comp[idx, 0] - min) / (max - min)
            draw_point(sph_pts[idx,], 0.1, (1,0,0), fig, alpha=alpha)
            alpha = (this_comp[idx, 1] - min) / (max - min)
            draw_point(sph_pts[idx,], 0.1, (0,1,0), fig, alpha=alpha)
            alpha = (this_comp[idx, 2] - min) / (max - min)
            draw_point(sph_pts[idx,], 0.1, (0,0,1), fig, alpha=alpha)
