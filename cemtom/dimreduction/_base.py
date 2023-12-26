from umap import UMAP as UMAPReducer
from sklearn.manifold import TSNE as TSNEReducer
from sklearn.decomposition import PCA as PCAReducer


class DimensionReductionBase:
    def __init__(self, **params):
        self.params = params

    def fit(self, X):
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        raise NotImplementedError("This method should be implemented by subclasses.")


class UMAP(DimensionReductionBase):
    def __init__(self,
                 n_neighbors=15,
                 n_components=2,
                 metric='euclidean',
                 **kwargs
                 ):
        super().__init__(n_neighbors=n_neighbors, n_components=n_components, metric=metric, **kwargs)
        self.model = UMAPReducer(self.params)

    def fit(self, X):
        return self.model.fit(X)

    def fit_transform(self, X):
        return self.model.fit_transform(X)


class TSNE(DimensionReductionBase):
    def __init__(self, n_components=2, perplexity=3, learning_rate='auto', **kwargs):
        super().__init__(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, **kwargs)
        self.model = TSNEReducer(**self.params)

    def fit_transform(self, data):
        return self.model.fit_transform(data)


class PCA(DimensionReductionBase):
    def __init__(self, n_components=2, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.model = PCAReducer(**self.params)

    def fit_transform(self, data):
        return self.model.fit_transform(data)


def select_dimension_reduction_model(model, **params):
    if model == "tsne":
        return TSNE(**params)
    if model == "pca":
        return PCA(**params)
    if model == "umap":
        return UMAP(**params)
    else:
        raise ValueError("Dimension reduction model does not exist")