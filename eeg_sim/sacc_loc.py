from os.path import isdir
import numpy as np
import mne
import matplotlib.pyplot as plt
import pickle
plt.ion()
from mayavi.mlab import points3d, plot3d, mesh, figure

def draw_pair(points, size, color, fig):
    points3d(points[:,0], points[:,1], points[:,2], scale_factor=size,
             color=color, figure=fig)
    plot3d(points[:,0], points[:,1], points[:,2],
           tube_radius=None, color=color, figure=fig)

def draw_eeg(info, size, color, fig):
    hsp = mne.bem.get_fitting_dig(info)
    for d_idx in range(len(hsp)):
        points3d(hsp[d_idx, 0], hsp[d_idx, 1], hsp[d_idx, 2], scale_factor=size,
                 color=color)

def draw_sphere(R, r0, color, alpha, fig):
    [phi, theta] = np.mgrid[0:2 * np.pi:12j, 0:np.pi:12j]
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return mesh(R * x + r0[0], R * y + r0[1], R * z + r0[2], color=color,
                opacity=alpha, figure=fig)

def expl_var_of_model(exg_rr, sphere):
    # random orientations; will be allowed to vary with localisation anyway
    exg_nn = np.random.randn(*exg_rr.shape)
    exg_nn = (exg_nn.T / np.linalg.norm(exg_nn, axis=1)).T

    src = mne.setup_volume_source_space(pos={"rr":exg_rr, "nn":exg_nn},
                                        sphere_units="mm")
    fwd = mne.make_forward_solution(evo.info, None, src, sphere, meg=False,
                                    eeg=True, n_jobs=2)
    evo.set_eeg_reference(projection=True)
    # covariance
    cov = mne.compute_covariance(epo)
    dipoles, residual = mne.beamformer.rap_music(evo, fwd, cov, n_dipoles=1,
                                 return_residual=True, verbose=True)
    # inv = mne.minimum_norm.make_inverse_operator(evo.info, fwd, cov,
    #                                              fixed=False, depth=0.8)

    # stc, resid, expl = mne.minimum_norm.apply_inverse(evo, inv, method="sLORETA",
    #                                                   pick_ori="vector",
    #                                                   return_residual=True)
    return dipoles, residual

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.
if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/reog/"

epo = mne.read_epochs("{}grand_saccade-epo.fif".format(eeg_dir))
epo.crop(tmin=-0.05, tmax=0.05)
evo = epo.average()

# fit sphere
R, r0, _ = mne.bem.fit_sphere_to_headshape(epo.info)
# make sphere
sphere = mne.bem.make_sphere_model(r0, head_radius=R,
                                   relative_radii=(0.97, 0.98, 0.99, 1.),
                                   sigmas=(0.33, 1.0, 0.004, 0.33), verbose=False)

# eyeball dipole locations
sine_dist = 2.6
exg_rr = np.array([[np.cos(np.pi / sine_dist), np.sin(np.pi /sine_dist), -0.6],
                   [-np.cos(np.pi / sine_dist), np.sin(np.pi / sine_dist), -0.6]])
exg_rr = np.random.normal(size=(5000,3))
exg_rr /= np.sqrt(np.sum(exg_rr**2, axis=1, keepdims=True))
exg_rr *= 0.96 * R
exg_rr += r0

dipoles, resid = expl_var_of_model(exg_rr, sphere)

# try:
#     with open("{}rand_dips.pickle".format(eeg_dir), "rb") as f:
#         res_dict = pickle.load(f)
#     expl_var_randos = res_dict["var"]
#     exg_rrs = res_dict["exg_rrs"]
# except:
#     expl_var_randos = []
#     exg_rrs = []
#     for iter_idx in range(1000):
#         # two random points on a sphere
#         exg_rr = np.random.normal(size=(2,3))
#         exg_rr /= np.sqrt(np.sum(exg_rr**2, axis=1, keepdims=True))
#         exg_rr *= 0.96 * R
#         exg_rr += r0
#
#         expl_var_randos.append(expl_var_of_model(exg_rr, sphere))
#         exg_rrs.append(exg_rr)
#         print("\n\n\nRando model explains {} "
#               "percent.\n\n\n".format(expl_var_randos[-1]))
#
#     expl_var_randos = np.array(expl_var_randos)
#     exg_rrs = np.array(exg_rrs)
#
#     res_dict = {"var":expl_var_randos, "exg_rrs":exg_rrs}
#     with open("{}rand_dips.pickle".format(eeg_dir), "wb") as f:
#         pickle.dump(res_dict, f)

fig = figure()
draw_eeg(epo.info, 0.01, (0,0,0), fig)
# get all exg that performed better than simple eye model
# better_inds = expl_var_randos > expl_var
# for bi in np.where(better_inds)[0]:
#     draw_pair(exg_rrs[bi,], 0.005, (1,0,0), fig)
exg_rr = []
for dip in dipoles:
    exg_rr.append(dip.pos[0])
draw_pair(np.array(exg_rr), 0.01, (0,0,1), fig)
draw_sphere(R, r0, (0,0,1), 0.1, fig)
