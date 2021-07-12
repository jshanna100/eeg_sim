from mat73 import loadmat
import matplotlib.pyplot as plt
import pickle
import numpy as np
plt.ion()

raw_dir = "/home/jev/Downloads/"
out_dir = "/home/jev/eeg_sim/mats/"

cnx_mat_fname = "averageConnectivity_Fpt.mat"
cnx_mat_var_fname = "connectivityCoefficientOfVariation.mat"
delay_mat_fname = "averageConnectivity_tractLengths.mat"

cnx = loadmat("{}{}".format(raw_dir, cnx_mat_fname))
cnx_var = loadmat("{}{}".format(raw_dir, cnx_mat_var_fname))
cnx_delay = loadmat("{}{}".format(raw_dir, delay_mat_fname))

# check that parcels match
if not cnx_delay["parcelIDs"] == cnx_var["parcelIDs"] == cnx["parcelIDs"]:
    raise ValueError("Parcel IDs do not match.")

cnx_norm = np.exp(cnx["Fpt"])
cnx_l = len(cnx_norm)
cnx_norm[np.arange(cnx_l), np.arange(cnx_l)] = np.zeros(cnx_l)

cnx_dict = {"cnx":cnx_norm, "cnx_var":cnx_var["coefficientOfVariation"],
            "cnx_delay":cnx_delay["tractLengths"],
            "Regions":cnx["parcelIDs"]}

with open("{}mats.pickle".format(out_dir), "wb") as f:
    pickle.dump(cnx_dict, f)
