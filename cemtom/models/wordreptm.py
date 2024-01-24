import numpy as np

from cemtom.models import CEBTMBase
from sklearn.cluster import KMeans
from ..clustering import SphericalKMeans, sort_closest_centers
from cemtom.embedder import BertTokenEmbedder
import umap
import hdbscan


class WordRepTM(CEBTMBase):
    def __init__(self,
                 nr_topics=20,
                 embedding_model: BertTokenEmbedder = None,
                 nr_dimensions=None,
                 cluster_model=None,
                 dataset="20news"):
        super().__init__(dataset=dataset)
        self.model_name = "wordreptm"
        self.nr_topics = nr_topics
        self.nr_dimensions = nr_dimensions
        self.clustering_model = cluster_model
        self.embedding_model = embedding_model
        if self.clustering_model is None:
            self.clustering_model = SphericalKMeans(n_clusters=self.nr_topics, init='k-means++')
        self.topk = None

    def fit_transform(self, documents):
        if self.embedding_model.embeddings_ is None:
            print("Embeddings not available...Generating the embeddings")
            self.embedding_model.embed(documents)
        embeddings = self._reduce_dimensionality(self.embedding_model.embeddings_)
        self._cluster_embeddings(embeddings)

    def _reduce_dimensionality(self, embeddings):
        if self.dim_reduction_model is not None:
            return self.dim_reduction_model.fit_transform(embeddings)
        return embeddings

    def _cluster_embeddings(self, embeddings):
        self.clustering_model.fit(embeddings)
        self.topk = sort_closest_centers(self.clustering_model.cluster_centers_, self.clustering_model.labels_,
                                         embeddings)

    def get_topic_words(self):
        embeds_vocab = np.array(self.embedding_model.embedding_vocab)
        topic_words = []
        for topic in range(self.nr_topics):
            topic_words.append(embeds_vocab[self.topk[topic]].tolist())
        return topic_words
