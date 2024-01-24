import bertopic
from ._cebtm import CEBTMBase
from bertopic import BERTopic


class Bertopic(CEBTMBase):
    def __init__(self):
        super().__init__()
        self.model_name = "bertopic"
        self.model = BERTopic(
            language=self.language,
            top_n_words=self.nr_top_words,
            min_topic_size=self.min_topic_size,
            nr_topics=self.nr_topics,
            embedding_model=self.embedding_model,
            umap_model=self.dim_reduction_model,
            hdbscan_model=self.clustering_model,
            vectorizer_model=self.vectorizer_model

        )

    def fit_transform(self, documents):
        self.model.fit_transform(documents)

    def get_topic_words(self):
        return self.model.get_topics()
