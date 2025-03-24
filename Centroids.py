# Interpretation of centroids

#**Author:** Luca A. Naudszus
#**Date:** 13 March 2025
#**Affiliation:** Social Brain Sciences Lab, ETH Zürich
#**Email:** luca.naudszus@gess.ethz.ch

# ------------------------------------------------------------
# Import packages and custom functions
import os
from pathlib import Path
#---
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Set path
os.chdir('/Users/lucanaudszus/Library/CloudStorage/OneDrive-Personal/Translational Neuroscience/9 Master Thesis/code')

# ------------------------------------------------------------
# Set variables
type_of_data = "one_brain"
exp_block_size = 4
if type_of_data == "one_brain":
    channels = ["LIFG", "LTPJ", "RIFG", "RTPJ"] 
else: 
    channels = ["LIFG_0", "LTPJ_0", "RIFG_0", "RTPJ_0",
                "LIFG_1", "LTPJ_1", "RIFG_1", "RTPJ_1"] 

# ------------------------------------------------------------
# Load data
path = Path("results")
fn_centroids = sorted(list(path.glob(f"cluster_means_{type_of_data}_*")))[-1]
centroids = np.load(fn_centroids)
fn_classes = sorted(list(path.glob(f"classes_{type_of_data}_*")))[-1]
classes = np.load(fn_classes)
fn_matrices = sorted(list(path.glob(f"matrices_{type_of_data}_*")))[-1]
matrices = np.load(fn_matrices)

# ------------------------------------------------------------
# Preprocess centroids

### Separate HbO and HbR data
centroids_hbo, centroids_hbr = [], []
for centroid in centroids:
    centroids_hbo.append(np.tril(centroid[:exp_block_size,:exp_block_size]))
    centroids_hbr.append(np.tril(centroid[exp_block_size:(exp_block_size*2),
                                         exp_block_size:(exp_block_size*2)]))

# ------------------------------------------------------------
# Plot

centroids = [centroids_hbo, centroids_hbr]
chromophores = ["HbO", "HbR"]
for chromophore in range(2):
    for in_centroid in range(len(centroids[chromophore])):
        fig, ax = plt.subplots()
        im = ax.imshow(centroids[chromophore][in_centroid])

        ax.set_xticks(range(len(channels)), labels=channels,
                rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(channels)), labels=channels)
        for i in range(len(channels)):
            for j in range(len(channels)):
                if j <= i:
                    text = ax.text(j, i, round(centroids[chromophore][in_centroid][i, j], 3),
                        ha="center", va="center", color="b")

        ax.set_title(f"Centroids {chromophores[chromophore]}, No.: {in_centroid + 1}")
        fig.tight_layout()
        plt.show()