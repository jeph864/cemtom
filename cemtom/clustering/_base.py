import hdbscan
from sklearn.cluster import AgglomerativeClustering as skAgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering as SpectralClus


class ClusteringBase:
    def __init__(self, **params):
        self.model = None
        self.params = params

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
    def __init__(self, n_clusters=8, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.model = KMeans(**self.params)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)


class GaussianMixtureClustering(ClusteringBase):
    def __init__(self, n_components=8, covariance_type='full', **kwargs):
        super().__init__(n_components=n_components, covariance_type=covariance_type, **kwargs)
        self.model = GaussianMixture(**self.params)

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)


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
