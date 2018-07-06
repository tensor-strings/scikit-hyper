"""
Agglomerative clustering
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering as _sklearn_agglomeration

from skhyper.process import Process


class Agglomeration:
    """

    """
    def __init__(self, n_clusters, init="agglomeration++", affinity='euclidean',
                 memory=None, connectivity=None, compute_full_tree='auto',
                 linkage='ward', pooling_func=np.mean):
        self._X = None

        self.mdl = None

        self.image_components_ = None
        self.spec_components_ = None
        self.n_leaves = None
        self.children_ = None

        self.labels_ = None

        self.n_clusters = n_clusters
        self.init = init
        self.affinity = affinity
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.pooling_func = pooling_func


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

        mdl = _sklearn_agglomeration(n_clusters=self.n_clusters, affinity=self.affinity, memory=self.memory,
                                     connectivity=self.connectivity, compute_full_tree=self.compute_full_tree,
                                     linkage=self.linkage, pooling_func=self.pooling_func).fit(self._X.flatten())

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
