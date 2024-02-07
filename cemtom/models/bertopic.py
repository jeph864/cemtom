import bertopic
from ._cebtm import CEBTMBase
from sentence_transformers import SentenceTransformer
#from umap import UMAP
from cemtom.dimreduction._base import UMAP
from cemtom.clustering._base import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import re


class Bertopic(CEBTMBase):
    def __init__(self, calculate_probabilities=True, **params):
        super().__init__(**params)
        self.model_name = "bertopic"
        self.model = None

        self.calculate_probabilities = calculate_probabilities
        self.doc_embeddings_ = None

    def fit_transform(self, documents, embeddings=None):
        def remove_numbers(text):
            text = text.lower()
            text = re.sub(r'\d+', '', text)
            return text

        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        if embeddings is None:
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        if self.dim_reduction_model is None:

            self.dim_reduction_model = UMAP(n_neighbors=10, n_components=self.nr_dimensions, min_dist=0.0,
                                            metric='euclidean',
                                            random_state=42)

        if self.clustering_model is None:
            self.clustering_model = HDBSCAN(min_cluster_size=30, metric='euclidean', cluster_selection_method='eom',
                                            prediction_data=True)
        self.vectorizer_model = CountVectorizer(stop_words="english", min_df=5, max_df=0.80, ngram_range=(1, 1)
                                                # ,token_pattern=r"(?u)\b\w{" + str(3) + ",}\b"
                                                , preprocessor=remove_numbers
                                                # , vocabulary=set([word for words in preprocessed_ds.get_corpus() for word in words.split()])
                                                )
        self.doc_embeddings_ = embeddings
        self.model = BERTopic(
            language=self.language,
            top_n_words=self.nr_top_words,
            min_topic_size=self.min_topic_size,
            nr_topics=self.nr_topics,
            embedding_model=self.embedding_model,
            umap_model=self.dim_reduction_model,
            hdbscan_model=self.clustering_model.model,
            vectorizer_model=self.vectorizer_model,
            calculate_probabilities=True,
            verbose=True,

        )
        return self.model.fit_transform(documents, embeddings=embeddings)

    def get_topic_words(self, topk=10, join=False):
        topics_words = []
        topics = self.model.get_topics()
        if -1 in topics:
            topics.pop(-1)
        for idx, topic in topics.items():
            top = [rep[0] for rep in topic if rep[0]]
            if len(top) >= topk:
                topics_words.append(top[:topk])
        if join:
            for idx, topic_words in enumerate(topics_words):
                topics_words[idx] = " ".join(topic_words)
        # for topic in
        return topics_words
