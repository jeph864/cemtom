import hdbscan
import numpy as np
import scipy as sp
from sklearn.cluster import AgglomerativeClustering as skAgglomerativeClustering
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering as SpectralClus


def _sort_closest_centers(centers, clusters, vocab_embedding, k=20):
    top_idx = []
    unique, counts = np.unique(clusters, return_counts=True)
    k = np.min((k, np.min(counts)))
    for topic in range(centers.shape[0]):
        diic = np.where(clusters == topic)[0]
        dist = np.sum((vocab_embedding[diic] - centers[topic]) ** 2, axis=1)
        topk = dist.argsort()[:k]
        top_idx = np.vstack((top_idx, diic[topk])) if topic > 0 else diic[topk]
    return top_idx


def __sort_dist2center(centers, clusters, vocab_embeddings, k):
    unique, counts = np.unique(clusters, return_counts=True)
    k = np.min((k, np.min(counts)))
    top_idx = []
    for c_ind in range(centers.shape[0]):
        data_idx_within_i_cluster = np.array([idx for idx, clu_num in enumerate(clusters) if clu_num == c_ind])
        one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster), centers.shape[1]))

        for row_num, data_idx in enumerate(data_idx_within_i_cluster):
            one_row = vocab_embeddings[data_idx]
            one_cluster_tf_matrix[row_num] = one_row

        dist_X = np.sum((one_cluster_tf_matrix - centers[c_ind]) ** 2, axis=1)

        topk_vals = dist_X.argsort().astype(int)
        top_idx.append(data_idx_within_i_cluster[topk_vals][:k])
    return np.vstack(top_idx)


class ClusteringBase:
    def __init__(self, **params):
        self.model = None
        self.params = params
        self.m_clusters = None

    def fit(self, data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def predict(self, data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_params(self):
        """
        Returns the parameters and hyperparameters of the clustering algorithm.
        """
        if self.model is not None:
            return self.model.get_params()


class HDBSCANClustering(ClusteringBase):
    def __init__(self, min_cluster_size=5, min_samples=None, **kwargs):
        super().__init__(min_cluster_size=min_cluster_size, min_samples=min_samples, **kwargs)
        self.model = hdbscan.HDBSCAN(**self.params)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.labels_


class DBSCANClustering(ClusteringBase):
    def __init__(self, eps=0.5, min_samples=5, **kwargs):
        super().__init__(eps=eps, min_samples=min_samples, **kwargs)
        self.model = DBSCAN(**self.params)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.fit_predict(data)


class KMeansClustering(ClusteringBase):
    def __init__(self, n_clusters=8, random_state=42, sort=True):
        super().__init__(n_clusters=n_clusters, random_state=random_state)
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.m_clusters = None
        self.sort = sort

    def fit(self, data, sample_weight=None):
        return self.model.fit(data, sample_weight=sample_weight)

    def transform(self, data, sample_weight=None):
        return self.model.predict(data, sample_weight=sample_weight)

    def fit_transform(self, data, sample_weight=None, k=200):
        self.fit(data, sample_weight=sample_weight)
        self.m_clusters = self.transform(data, sample_weight)
        return KMeansClustering.__sort_dist2center(self.model.cluster_centers_, self.m_clusters, data, k)

    def get_sorted(self, embeddings, k=10):
        return KMeansClustering.__sort_dist2center(self.model.cluster_centers_, self.m_clusters, embeddings, k)

    @staticmethod
    def __sort_dist2center(centers, clusters, vocab_embeddings, k):
        unique, counts = np.unique(clusters, return_counts=True)
        k = np.min((k, np.min(counts)))
        top_idx = []
        for c_ind in range(centers.shape[0]):
            data_idx_within_i_cluster = np.array([idx for idx, clu_num in enumerate(clusters) if clu_num == c_ind])
            one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster), centers.shape[1]))

            for row_num, data_idx in enumerate(data_idx_within_i_cluster):
                one_row = vocab_embeddings[data_idx]
                one_cluster_tf_matrix[row_num] = one_row

            dist_X = np.sum((one_cluster_tf_matrix - centers[c_ind]) ** 2, axis=1)

            topk_vals = dist_X.argsort().astype(int)
            top_idx.append(data_idx_within_i_cluster[topk_vals][:k])
        return np.vstack(top_idx)


class GaussianMixture(ClusteringBase):
    def __init__(self, n_components=8, covariance_type='full', random_state=42):
        super().__init__(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
        self.model = GMM(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
        self.topk_sorted = None

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)

    def fit_transform(self, data, sample_weight=None, k=200):
        self.model.fit(data)
        topk = []
        for i in range(self.model.n_components):
            density = sp.stats.multivariate_normal(cov=self.model.covariances_[i], mean=self.model.means_[i]).logpdf(data)
            top_idx = density.argsort()[-1 * len(density):][::-1].astype(int)
            topk.append(top_idx)
        self.topk_sorted = topk
        self.m_clusters = self.model.predict(data)
        return np.vstack([topic[:k] for topic in self.topk_sorted])

    def get_sorted(self, embeddings=None, k=10):
        return np.vstack([topic[:k] for topic in self.topk_sorted])


class OPTICSClustering(ClusteringBase):
    def __init__(self, min_samples=5, max_eps=np.inf, **kwargs):
        super().__init__(min_samples=min_samples, max_eps=max_eps, **kwargs)
        self.model = OPTICS(**self.params)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.fit_predict(data)


class AgglomerativeClustering(ClusteringBase):
    def __init__(self, n_clusters=8, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.model = skAgglomerativeClustering(**self.params)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.fit_predict(data)


class SpectralClustering(ClusteringBase):
    def __init__(self, n_clusters=8, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.model = SpectralClus(**self.params)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.fit_predict(data)


def select_clustering(model, **params):
    pass
