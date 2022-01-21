from os.path import isdir
import numpy as np
import mne
from mne.simulation.raw import _SimForwards, _check_head_pos
from mne.minimum_norm.inverse import (_check_or_prepare, _assemble_kernel,
                                      _pick_channels_inverse_operator)
import matplotlib.pyplot as plt
import pickle
plt.ion()
from mayavi import mlab
from mayavi.mlab import points3d, plot3d, mesh, quiver3d, figure
from utils import *
from itertools import product

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/reog/"
mat_dir = base_dir+"eeg_sim/mats/"
img_dir = eeg_dir+"anim/"

loc_time = (-0.025, 0.025)
loc_time = (-0.01, 0.01)
loc_time = (-0.02, 0.05)
animate = False

snr = 3.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

try:
    evo = mne.read_evokeds("{}grand_saccade-ave.fif".format(eeg_dir))[0]
    cov = mne.read_cov("{}grand_saccade-cov.fif".format(eeg_dir))
except:
    epo = mne.read_epochs("{}grand_saccade-epo.fif".format(eeg_dir))
    epo.set_eeg_reference(projection=True)
    epo.apply_proj()
    cov = mne.compute_covariance(epo.copy().crop(tmin=epo.times[0], tmax=-0.05))
    evo = epo.copy().average(picks="all")
    evo.save("{}grand_saccade-ave.fif".format(eeg_dir))
    cov.save("{}grand_saccade-cov.fif".format(eeg_dir))

epo = mne.read_epochs("{}grand_saccade-epo.fif".format(eeg_dir))
evo.crop(tmin=loc_time[0], tmax=loc_time[1])
epo.crop(tmin=loc_time[0], tmax=loc_time[1])

info, times, first_samp = evo.info, evo.times, 0

# fit sphere
R, r0, _ = mne.bem.fit_sphere_to_headshape(evo.info)
# make sphere
sphere = mne.bem.make_sphere_model(r0, head_radius=R,
                                   relative_radii=(0.97, 0.98, 0.99, 1.),
                                   sigmas=(0.33, 1.0, 0.004, 0.33), verbose=False)


### iterate through different eye locations to find best postion
sine_dists = np.linspace(2.266, 3.2, 10)
Zs = np.linspace(-1.2, -.4, 10)
eye_infos = list(product(sine_dists, Zs))

expl_vars = []
exg_rrs = []
for eye_info in eye_infos:
    sd, Z = eye_info
    sine_dist = 2.6
    exg_rr = np.array([[np.cos(np.pi / sd), np.sin(np.pi / sd), Z],
                       [-np.cos(np.pi / sd), np.sin(np.pi / sd), Z]])
    exg_rr /= np.sqrt(np.sum(exg_rr**2, axis=1, keepdims=True))
    exg_rr *= 0.96 * R
    exg_rr += r0

    # random orientations; will be allowed to vary with localisation anyway
    exg_nn = np.random.randn(*exg_rr.shape)
    exg_nn = (exg_nn.T / np.linalg.norm(exg_nn, axis=1)).T

    src = mne.setup_volume_source_space(pos={"rr":exg_rr, "nn":exg_nn},
                                        sphere_units="mm")
    dev_head_ts, offsets = _check_head_pos(None, info, first_samp, times)
    get_fwd = _SimForwards(dev_head_ts, offsets, info, None, src, sphere, 0.005,
                           8, mne.pick_types(info, eeg=True))
    fwd = next(get_fwd.iter)
    inv = mne.minimum_norm.make_inverse_operator(epo.info, fwd, cov,
                                                 fixed=False, depth=0.8)
    stc, resid, expl = mne.minimum_norm.apply_inverse(evo, inv, method="MNE",
                                                      pick_ori="vector",
                                                      return_residual=True)

    print("\n\n\nEye variance: {}\n\n\n".format(expl))
    expl_vars.append(expl)
    exg_rrs.append(exg_rr)
expl_vars = np.array(expl_vars)
print("\n\n\nBest Eye variance: {}\n\n\n".format(expl_vars.max()))
best_rr = exg_rrs[expl_vars.argmax()]

fig = mlab.figure()
draw_sphere(R, r0, (0,0,1), 0.5, fig)
draw_eeg(evo.info, 0.01, (0,0,0), fig)
draw_pair(best_rr, 0.01, (1,0,0), fig)

draw_pair(exg_rrs[0], 0.01, (0,1,0), fig)
draw_pair(exg_rrs[-1], 0.01, (1,1,0), fig)

breakpoint()

### localise epochs
# eyeball dipole locations
sine_dist = 2.58
Z = -0.489
exg_rr = np.array([[np.cos(np.pi / sine_dist), np.sin(np.pi / sine_dist), Z],
                   [-np.cos(np.pi / sine_dist), np.sin(np.pi / sine_dist), Z]])
exg_rr /= np.sqrt(np.sum(exg_rr**2, axis=1, keepdims=True))
exg_rr *= 0.96 * R
exg_rr += r0

# random orientations; will be allowed to vary with localisation anyway
exg_nn = np.random.randn(*exg_rr.shape)
exg_nn = (exg_nn.T / np.linalg.norm(exg_nn, axis=1)).T

src = mne.setup_volume_source_space(pos={"rr":exg_rr, "nn":exg_nn},
                                    sphere_units="mm")
dev_head_ts, offsets = _check_head_pos(None, info, first_samp, times)
get_fwd = _SimForwards(dev_head_ts, offsets, info, None, src, sphere, 0.005,
                       8, mne.pick_types(info, eeg=True))
fwd = next(get_fwd.iter)
inv = mne.minimum_norm.make_inverse_operator(epo.info, fwd, cov,
                                             fixed=False, depth=0.8)

stcs = mne.minimum_norm.apply_inverse_epochs(epo, inv, lambda2, method="MNE",
                                             pick_ori="vector")
### localise random two points iteratively
# expls = []
# rand_rrs = []
# for shuf_idx in range(0):
#     r_rr = get_rand_rrs((2,3), R, r0, z_excl=0.)
#
#     # random orientations; will be allowed to vary with localisation anyway
#     r_nn = np.random.randn(*r_rr.shape)
#     r_nn = (r_nn.T / np.linalg.norm(r_nn, axis=1)).T
#
#     src = mne.setup_volume_source_space(pos={"rr":r_rr, "nn":r_nn},
#                                         sphere_units="mm")
#     dev_head_ts, offsets = _check_head_pos(None, info, first_samp, times)
#     get_fwd = _SimForwards(dev_head_ts, offsets, info, None, src, sphere, 0.005,
#                            8, mne.pick_types(info, eeg=True))
#     fwd = next(get_fwd.iter)
#     inv = mne.minimum_norm.make_inverse_operator(evo.info, fwd, cov,
#                                                  fixed=False, depth=0.8)
#
#     r_stc, r_resid, r_expl = mne.minimum_norm.apply_inverse(evo, inv, method="MNE",
#                                                             pick_ori="vector",
#                                                             return_residual=True)
#     print("\n\n\nRando variance: {}\n\n\n".format(r_expl))
#     expls.append(r_expl)
#     rand_rrs.append(r_rr)
# expls = np.array(expls)
# rand_rrs = np.array(rand_rrs)



## save dipoles

vecs = np.zeros((len(epo), *stcs[0].data.shape))
for epo_idx in range(len(epo)):
     vecs[epo_idx,] = stcs[epo_idx].data

a = vecs.reshape(len(vecs), 6, vecs.shape[-1])
b = np.hstack(a)
np.save("{}dipole_vecs.npy".format(mat_dir), vecs)

## draw sphere and dipoles
# fig = figure()
# draw_eeg(evo.info, 0.01, (0,0,0), fig)
# draw_sphere(R, r0, (0,0,1), 0.1, fig)
# # get all rrs that performed better than simple eye model
# better_inds = expls > expl
# for bi in np.where(better_inds)[0]:
#     draw_pair(rand_rrs[bi,], 0.005, (1,0,0), fig)
# draw_pair(exg_rr, 0.01, (0,0,1), fig)
