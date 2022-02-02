from os import listdir
from os.path import isdir
import mne
from mne.viz import plot_topomap
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import pandas as pd
plt.ion()
import umap
import matplotlib.colors as colors
import pickle
import hdbscan

class Manifold():
    def __init__(self, data, fig, ax, dimen="2d", interact=True, cmap="PuRd",
                 cat_cmap=plt.cm.tab20, marksize=1, pickradius=2,
                 col_dict=None, df=None, label_with_subject=None,
                 background=(0,0,0)):
        self.data = data
        self.fig = fig
        self.ax = ax
        self.dimen = dimen
        self.interact = interact
        self.cmap = cmap
        self.cat_cmap = cat_cmap
        self.marksize = marksize
        self.pickradius = pickradius
        self.col_dict = col_dict
        if df is not None:
            self.subjs = df["Subj"]
            self.subj_unq = list(df["Subj"].unique())
        self.df = df
        self.label_with_subject = label_with_subject
        self.background = background

        self.subj_mode = False
        self.cur_subj = None
        self.active_inds = None

    def next(self):
        if self.cur_subj is None:
            self.draw(self.subj_unq[0])
        elif self.subj_unq.index(self.cur_subj) == (len(self.subj_unq)-1):
            print("No more subjects.")
            return
        else:
            self.draw(self.subj_unq[self.subj_unq.index(self.cur_subj)+1])

    def prev(self):
        if self.cur_subj is None:
            self.draw(self.subj_unq[0])
        elif self.subj_unq.index(self.cur_subj) == 0:
            print("No more subjects.")
            return
        else:
            self.draw(self.subj_unq[self.subj_unq.index(self.cur_subj)-1])

    def draw(self, label):
        self.ax.clear()
        self.ax.set_facecolor(self.background)
        if label in self.col_dict.keys():
            # preprogrammed colour set
            cols = self.col_dict[label]
            if self.dimen == "2d":
                self.ax.scatter(self.data[:,0], self.data[:,1], c=cols,
                                cmap=self.cmap, s=self.marksize,
                                picker=self.interact,
                                pickradius=self.pickradius)
            if self.dimen == "3d":
                self.ax.scatter(self.data[:,0], self.data[:,1], self.data[:,2],
                                c=cols, cmap=self.cmap, s=self.marksize,
                                picker=self.interact,
                                pickradius=self.pickradius)
            self.subj_mode = False
            self.active_inds = np.arange(len(self.data))
        elif self.subjs is not None and label in self.subj_unq:
            ## highlight a subject
            cols = np.ones((len(self.data), 4)) * 0.05
            row_inds = np.where(self.df["Subj"] == label)[0]
            else_inds = np.where(self.df["Subj"] != label)[0]
            var_expls = unit_range(np.log(self.df.iloc[row_inds]["VarExpl"].values))
            for abs_idx, row_idx in enumerate(row_inds):
                # categorical colour
                if self.label_with_subject is None:
                    # colour by chunk
                    cols[row_idx,] = self.cat_cmap(self.df.iloc[row_idx]["Chunk"])
                    cols[row_idx, 3] = var_expls[abs_idx]
                else:
                    # colour by value of subject_label
                    cols[row_idx,] = self.col_dict[self.label_with_subject][row_idx,]

            if self.dimen == "2d":
                self.ax.scatter(self.data[row_inds,0], self.data[row_inds,1],
                                c=cols[row_inds], cmap=self.cmap,
                                s=25,
                                picker=self.interact,
                                pickradius=self.pickradius)
                self.ax.scatter(self.data[else_inds,0], self.data[else_inds,1],
                                c=cols[else_inds], cmap=self.cmap,
                                s=self.marksize)
            if self.dimen == "3d":
                self.ax.scatter(self.data[row_inds,0], self.data[row_inds,1],
                                self.data[row_inds,2], c=cols[row_inds],
                                cmap=self.cmap, s=25,
                                picker=self.interact,
                                pickradius=self.pickradius)
                self.ax.scatter(self.data[else_inds,0], self.data[else_inds,1],
                                self.data[else_inds,2], c=cols[else_inds],
                                cmap=self.cmap, s=self.marksize)
            self.ax.set_title(label)

            self.subj_mode = True
            self.cur_subj = label
            self.active_inds = row_inds
        else:
            raise ValueError("No viable label.")

        if self.interact:
            self.fig.canvas.mpl_connect("pick_event", onpick)

        if self.subj_mode and self.label_with_subject is None:
            lines = [plt.Line2D([0], [0], color=self.cat_cmap(x), lw=4)
                     for x in range(20)]
            self.ax.legend(lines, ["Chunk {}".format(x) for x in np.arange(20)])

    def onpick(self, event_ind):
        for abs_idx, ica_idx in enumerate(event_ind):
            row = df.iloc[self.active_inds[ica_idx],]
            comp_vec = np.array([row[ch] for ch in inst.ch_names])
            fig, axes = plt.subplots(1, 2, figsize=(19.2, 19.2))
            plot_topomap(comp_vec, inst.info, axes=axes[0])
            band_vec =  np.array([row[band] for band in ["delta", "theta", "alpha",
                                                         "beta", "low_gamma",
                                                         "high_gamma"]])
            axes[1].bar(np.arange(6), band_vec)
            axes[1].set_xticks(np.arange(6),
                               labels=["delta", "theta", "alpha", "beta",
                                       "low_gamma", "high_gamma"])
            axes[0].set_title("({}) {}, {:.3f}, Chunk {}".format(abs_idx,
                                                                 row["Subj"],
                                                                 row["VarExpl"],
                                                                 row["Chunk"]))
            axes[1].set_ylim((0, 1))
            plt.tight_layout()

        # plot component sources
        this_df = df.iloc[event_ind]
        loc_inds, src_inds = this_df["LocIdx"].values, this_df["SrcIdx"].values
        srcs = []
        for loc_idx, src_idx in zip(loc_inds, src_inds):
            srcs.append(sources[src_idx][loc_idx,])
        srcs = np.array(srcs)
        info = mne.create_info(len(srcs), inst.info["sfreq"])
        raw = mne.io.RawArray(srcs, info)
        raw.plot(scalings={"misc":10}, duration=5)

class MplColorHelper:
  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

def unit_range(x):
    return (x - x.min()) / (x.max() - x.min())

def onpick(event):
    print(event.ind)
    inter_obj.onpick(event.ind)

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", action="store_true")
parser.add_argument("--label", type=str, default="cluster")
opt = parser.parse_args()

n_nx = 100
min_dist = .2

marksize = 2

# Define directory. The if statements are here because I work on multiple
# computers, and these automatically detect which one I'm using.
if isdir("/home/jev"):
    base_dir = "/home/jev/"
elif isdir("/home/hannaj/"):
    base_dir = "/home/hannaj/"
proc_dir = base_dir + "hdd/memtacs/proc/reog/"

# use this as a channel template
inst = mne.read_evokeds("{}grand_saccade-ave.fif".format(proc_dir))[0]
chan_dict = {"Vo":"eog","Vu":"eog","Re":"eog","Li":"eog"}
inst.set_channel_types(chan_dict)
inst = inst.pick_types(eeg=True)

df = pd.read_pickle("{}comp_vecs.pickle".format(proc_dir))
df_chs = list(df.columns[8:-6])
if opt.gamma:
    df = df[df["Gamma"]==True]
else:
    df = df[df["Gamma"]==False]

data = df.iloc[:, 8:]

# get component sources
print("Loading component sources.")
sources = np.load("{}comp_sources.npy".format(proc_dir), allow_pickle=True)

print("UMAP 3d")
reducer3d = umap.UMAP(n_components=3, n_neighbors=n_nx, min_dist=min_dist)
trans3d = reducer3d.fit_transform(data.abs())

print("UMAP 2d")
reducer2d = umap.UMAP(n_components=2, n_neighbors=n_nx, min_dist=min_dist)
trans2d = reducer2d.fit_transform(data.abs())

### set up the colour coding

# by saccade template match
print("Calculating template matches.")
twin = [-0.01, 0]
twin_inds = inst.time_as_index(twin)
template = inst.data[:, twin_inds[0]:twin_inds[1]].mean(axis=1)
# make sure channel ordering is correct
chan_inds = np.array([inst.ch_names.index(ch) for ch in df_chs])
template = template[chan_inds, ]
template = abs(template) # we are only interested in magnitude, not polarity
# 0 to 1 range
template = unit_range(template)
# get comp data for all rows
comps = abs(df.iloc[:, 8:-6].values)
matches = np.dot(comps, template)
# 0 to 1 range
temp_match = unit_range(matches)

# by variance explained
print("Calculating Variance Explained Ranks.")
var_expl = df["VarExpl"].values
var_expl = np.log(var_expl)
var_expl_rank = unit_range(var_expl)

## by spectral properties
print("Spectral colours.")
colget = MplColorHelper("Set3", 0, 12)
spect_cols = np.zeros((len(df), 4), dtype=np.float64)
band_data = df.iloc[:, -6:].values
band_data -= band_data.mean(axis=0) # mean centre
# normalise to 0-1
for col_idx in range(band_data.shape[1]):
    this_col = band_data[:, col_idx]
    band_data[:, col_idx] = unit_range(this_col)
# each row sums to 1
band_data = np.dot(np.diag(1/band_data.sum(axis=1)), band_data)
# assign weighted colour
for row_idx in range(len(band_data)):
    for band_idx, band in enumerate(band_data[row_idx,]):
        spect_cols[row_idx,] += colget.get_rgb(band_idx) * np.ones(4) * band
    spect_cols[spect_cols>1.] = 1


# by hdbscan
print("HDBSCAN clustering.")
labs = hdbscan.HDBSCAN(min_cluster_size=100,
                       min_samples=1).fit_predict(trans3d)
h_cols = np.zeros((len(labs), 4))
col_map = np.linspace(0., .8, labs.max()+1)
for lab_idx, lab in enumerate(labs):
    if lab == -1:
        h_cols[lab_idx,] = np.ones(4) * 0.05
    else:
        h_cols[lab_idx,] = plt.cm.hsv(col_map[lab])

col_dict = {"template":temp_match, "varexpl":var_expl_rank,
            "spectral":spect_cols, "hdbscan":h_cols}

fig2d, ax2d = plt.subplots()
fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection='3d')

man2d = Manifold(trans2d, fig2d, ax2d, col_dict=col_dict, df=df,
                 marksize=marksize, label_with_subject="hdbscan",
                 cat_cmap=plt.cm.Set3, cmap="PuRd")
man2d.draw("hdbscan")
inter_obj = man2d

man3d = Manifold(trans3d, fig3d, ax3d, dimen="3d", col_dict=col_dict, df=df,
                 marksize=marksize, interact=False, cat_cmap=plt.cm.Set3,
                 cmap="PuRd")
man3d.draw("hdbscan")
