import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(
    n_samples=10000,
    n_features=3,
    centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
    cluster_std=[0.2, 0.1, 0.2, 0.2],
    random_state=9,
)
fig = plt.figure()
plt.scatter(X_new[:, 0], X_new[:, 1], marker="o")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(
    n_samples=10000,
    n_features=3,
    centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
    cluster_std=[0.2, 0.1, 0.2, 0.2],
    random_state=9,
)
fig = plt.figure()
plt.scatter(X_new[:, 0], X_new[:, 1], marker="o")
plt.show()
