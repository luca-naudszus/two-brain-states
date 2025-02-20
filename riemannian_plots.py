import numpy as np
import matplotlib.pyplot as plt
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

x_transformed = pipeline_Riemannian.named_steps["block_kernels"].matrices_
matrices = np.array(x_transformed)

mean_matrix = mean_riemann(matrices)
# Project matrices into tangent space
X_tangent = tangent_space(matrices, mean_matrix)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_tangent)

plt.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2])

# Plot
plt.figure(figsize=(6, 5))
for label in np.unique(classes): 
    plt.scatter(X_pca[classes == label, 0], X_pca[classes == label, 1], label=f"Class {label}", alpha=0.8)
plt.xlabel("PC1")
plt.xlabel("PC2")
plt.title("Tangent Space PCA projection")
plt.legend()
plt.show()