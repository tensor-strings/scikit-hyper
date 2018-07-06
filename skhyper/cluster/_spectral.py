"""
Spectral clustering
"""
import numpy as np
from sklearn.cluster import SpectralClustering as _sklearn_spectral

from skhyper.process import Process


class Spectral:
    """
    Spectral clustering

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.
    """
    def __init__(self, n_clusters, init="spectral++", eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf',
                 n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3,
                 coef0=1, kernel_params=None, n_jobs=1):
        self._X = None

        self.mdl = None

        self.image_components_ = None
        self.spec_components_ = None

        self.labels_ = None
        # self.inertia_ = None

        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs


    def fit(self, X):
        """
        Creates affinity matrix for the hyperspectral image using specified affinity,
        and applies spectral clustering to affinity matrix

        :param X: object, type (Process)

        :return: self : Object
        """
        self._X = X
        if not isinstance(self._X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        mdl = _sklearn_spectral(n_clusters=self.n_clusters, eigen_solver=self.eigen_solver, random_state=self.random_state,
                                n_init=self.n_init, gamma=self.gamma, affinity=self.affinity,
                                n_neighbors=self.n_neighbors, eigen_tol=self.eigen_tol, degree=self.degree,
                                kernel_params=self.kernel_params, coef0=self.coef0, n_jobs=self.n_jobs).fit(self._X.flatten())

        labels = np.reshape(mdl.labels_, self._X.shape[:-1])
        labels += 1

        self.mdl = mdl
        self.labels_ = labels
        # self.inertia_ = mdl.inertia_

        self.image_components_, self.spec_components_ = [0]*self.n_clusters, [0]*self.n_clusters
        for cluster in range(self.n_clusters):
            self.image_components_[cluster] = np.squeeze(np.where(labels == cluster + 1, labels, 0) / (cluster + 1))

            self.spec_components_[cluster] = np.zeros(self._X.shape)
            for spectral_point in range(self._X.shape[-1]):
                self.spec_components_[cluster][..., spectral_point] = np.multiply(self._X[..., spectral_point],
                                                                                  np.where(labels == cluster + 1, labels, 0) / (cluster + 1))

            if self._X.n_dimension == 3:
                self.spec_components_[cluster] = np.squeeze(np.mean(np.mean(self.spec_components_[cluster], 1), 0))

            elif self._X.n_dimension == 4:
                self.spec_components_[cluster] = np.squeeze(np.mean(np.mean(np.mean(self.spec_components_[cluster], 2), 1), 0))

        return self
