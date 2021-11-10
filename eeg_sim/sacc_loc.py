from os.path import isdir
import numpy as np
import mne
from mne.simulation.raw import _SimForwards, _check_head_pos
import matplotlib.pyplot as plt
import pickle
plt.ion()
from mayavi.mlab import points3d, plot3d, mesh, figure

def get_rand_rrs(size, R, r0, z_excl=0.):
    r_rr = np.zeros(size)
    for r_idx in range(size[0]):
        r_r = np.random.normal(size=(1, size[1]))
        r_r /= np.sqrt(np.sum(r_r**2, axis=1, keepdims=True))
        r_r *= 0.96 * R
        r_r += r0
        while r_r[0, -1] < z_excl:
            r_r = np.random.normal(size=(1, size[1]))
            r_r /= np.sqrt(np.sum(r_r**2, axis=1, keepdims=True))
            r_r *= 0.96 * R
            r_r += r0
        r_rr[r_idx,] = r_r
    return r_rr

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

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.

if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
eeg_dir = base_dir+"hdd/memtacs/proc/reog/"

loc_time = (-0.0125, 0.0125)
loc_time = (-0.006, -0.002)
# loc_time = (0.002, 0.006)
loc_time = (-0.01, 0.01)
loc_resid = False

try:
    evo = mne.read_evokeds("{}grand_saccade-ave.fif".format(eeg_dir))[0]
    cov = mne.read_cov("{}grand_saccade-cov.fif".format(eeg_dir))
except:
    epo = mne.read_epochs("{}grand_saccade-epo.fif".format(eeg_dir))
    epo.set_eeg_reference(projection=True)
    epo.apply_proj()
    cov = mne.compute_covariance(epo.copy().crop(tmin=epo.times[0], tmax=-0.1))
    evo = epo.copy().average(picks="all")
    evo.save("{}grand_saccade-ave.fif".format(eeg_dir))
    cov.save("{}grand_saccade-cov.fif".format(eeg_dir))

evo.crop(tmin=loc_time[0], tmax=loc_time[1])

info, times, first_samp = evo.info, evo.times, 0

# fit sphere
R, r0, _ = mne.bem.fit_sphere_to_headshape(evo.info)
# make sphere
sphere = mne.bem.make_sphere_model(r0, head_radius=R,
                                   relative_radii=(0.97, 0.98, 0.99, 1.),
                                   sigmas=(0.33, 1.0, 0.004, 0.33), verbose=False)

# eyeball dipole locations
sine_dist = 2.6
exg_rr = np.array([[np.cos(np.pi / sine_dist), np.sin(np.pi /sine_dist), -0.6],
                   [-np.cos(np.pi / sine_dist), np.sin(np.pi / sine_dist), -0.6]])
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
inv = mne.minimum_norm.make_inverse_operator(evo.info, fwd, cov,
                                             fixed=False, depth=0.8)

stc, resid, expl = mne.minimum_norm.apply_inverse(evo, inv, method="sLORETA",
                                                  pick_ori="vector",
                                                  return_residual=True)
print("\n\n\nEye variance: {}\n\n\n".format(expl))

expls = []
rand_rrs = []
if loc_resid:
    this_evo = resid
else:
    this_evo = evo

for shuf_idx in range(0):
    r_rr = get_rand_rrs((2,3), R, r0, z_excl=0.)

    # random orientations; will be allowed to vary with localisation anyway
    r_nn = np.random.randn(*r_rr.shape)
    r_nn = (r_nn.T / np.linalg.norm(r_nn, axis=1)).T

    src = mne.setup_volume_source_space(pos={"rr":r_rr, "nn":r_nn},
                                        sphere_units="mm")
    dev_head_ts, offsets = _check_head_pos(None, info, first_samp, times)
    get_fwd = _SimForwards(dev_head_ts, offsets, info, None, src, sphere, 0.005,
                           8, mne.pick_types(info, eeg=True))
    fwd = next(get_fwd.iter)
    inv = mne.minimum_norm.make_inverse_operator(this_evo.info, fwd, cov,
                                                 fixed=False, depth=0.8)

    r_stc, r_resid, r_expl = mne.minimum_norm.apply_inverse(this_evo, inv, method="sLORETA",
                                                          pick_ori="vector",
                                                          return_residual=True)
    print("\n\n\nRando variance: {}\n\n\n".format(r_expl))
    expls.append(r_expl)
    rand_rrs.append(r_rr)

expls = np.array(expls)
rand_rrs = np.array(rand_rrs)

fig = figure()
draw_eeg(evo.info, 0.01, (0,0,0), fig)
# get all rrs that performed better than simple eye model
better_inds = expls > expl
for bi in np.where(better_inds)[0]:
    draw_pair(rand_rrs[bi,], 0.005, (1,0,0), fig)

draw_pair(exg_rr, 0.01, (0,0,1), fig)
draw_sphere(R, r0, (0,0,1), 0.1, fig)
