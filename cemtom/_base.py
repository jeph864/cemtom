import abc

from cemtom.clustering import ClusteringBase
from cemtom.dataset import fetch_dataset, Dictionary
from cemtom.preprocessing import Preprocessor as Pipe
from cemtom.embedder.word_embedder import get_embeddings
from cemtom.models import Sia
from cemtom.models.bertopic import Bertopic
from cemtom.dimreduction._base import (UMAP)
from cemtom.clustering._base import (HDBSCAN, KMeansClustering, GaussianMixture)
import argparse
from cemtom.utils.save_utils import save_embeddings

pretrained_models = 'pretrained_models'


class Trainer:
    def __init__(self, model='sia', vocab_embeddings='bert',
                 nr_topics=20,
                 nr_dimensions=5,
                 min_topic_size=10,
                 nr_top_words=10,
                 ###
                 sia_weighting=None,
                 sia_rerank=None,
                 vocab_embeddings_path=None,
                 pretrained_models_path=pretrained_models,
                 dataset=None,
                 preprocessor=None,
                 embedding_model=None
                 ):
        self.model_name = model

        # General topic variables
        self.nr_topics = nr_topics
        self.nr_dimensions = nr_dimensions
        self.min_topic_size = min_topic_size
        self.nr_top_words = nr_top_words

        ## Sia
        self.sia_weighting = sia_weighting
        self.sia_rerank = sia_rerank
        self.vocab_embeddings = vocab_embeddings
        self.pretrained_models_path = pretrained_models_path
        self.vocab_embeddings_path = vocab_embeddings_path

        self.preprocessor = preprocessor

        self.dataset = dataset
        if self.preprocessor is None or self.dataset is None:
            print("Loading default dataset, since the preprocessor and the dataset are empty")
            self.preprocessor, self.dataset = get_data()
        self.training_output = {
            'topics': None,
            'test-topic-document-matrix': None,
            'topic-document-matrix': None
        }
        self.dictionary = None

    def train(self):
        pass

    def evaluate(self):
        pass

    def train_sia(self, preprocessor=None):

        if preprocessor is None:
            preprocessor = self.preprocessor
        vocab_embeddings, vocabulary, word2idx, embedding_model = get_embeddings(name=self.vocab_embeddings,
                                                                                 path=self.vocab_embeddings_path,
                                                                                 vocabulary=preprocessor.vocabulary)
        if self.vocab_embeddings == "bert":
            print(f"Using Bert Embeddings. Vocabulary: {len(vocabulary)}")
            preprocessor, data = get_data(self.dataset.name, vocabulary=vocabulary)

        if self.vocab_embeddings == "fasttext":
            pass
        else:
            pass
        train_corpus, test_corpus = preprocessor.data.get_partitioned()
        corpus = train_corpus + test_corpus
        dictionary = Dictionary([doc.split() for doc in corpus], vocabulary)
        model = Sia(vocab=vocabulary, embedding_model_name=self.vocab_embeddings, vectorizer=preprocessor.vectorizer,
                    nr_dimensions=self.nr_dimensions,
                    nr_topics=self.nr_topics,
                    weighting=self.sia_weighting,
                    rerank=self.sia_rerank,
                    word2idx=word2idx
                    )
        model.fit_transform(corpus, word2doc=dictionary.word2doc, embeddings=vocab_embeddings)
        model.save_topics(path=f"sia_{self.preprocessor.data.name}_{self.vocab_embeddings}_topics.json")
        self.training_output['topics'] = model.get_topic_words()
        self.dictionary = dictionary

    def train_bertopic(self, embedding_model=None, embeddings=None):

        train_corpus, test_corpus = self.dataset.get_partitioned()
        corpus = train_corpus + test_corpus
        clustering_algo = get_clustering_model(algo='hdbscan')

        model = Bertopic(calculate_probabilities=True,
                         language='english',
                         nr_top_words=self.nr_top_words,
                         min_topic_size=self.min_topic_size,
                         nr_topics=self.nr_topics,
                         nr_dimensions=self.nr_dimensions,
                         embedding_model=embedding_model,
                         clustering_model=clustering_algo
                         )
        topics, probs = model.fit_transform(corpus, embeddings=embeddings)

        model.save_topics(path=f"bertopic_{self.dataset.name}_topics.json")

        self.training_output['topics'] = model.get_topic_words()
        topic_doc_dist, _ = model.model.approximate_distribution(train_corpus, window=3,
                                                                 use_embedding_model=False)
        test_topic_doc_dist, _ = model.model.approximate_distribution(test_corpus, window=3,
                                                                      use_embedding_model=False)

        self.training_output['topic-document-matrix'] = topic_doc_dist.T
        self.training_output['test-topic-document-matrix'] = test_topic_doc_dist.T
        return model


def get_data(dataset='20NewsGroup', vocabulary=None):
    token_dict = {
        "doc_start_token": '<s>',
        "doc_end_token": '</s>',
        "unk_token": '<unk>',
        "email_token": '<email>',
        "url_token": '<url>',
        "number_token": '<number>',
        "alpha_num_token": '<alpha_num>'
    }
    min_chars = 4
    data = None
    if dataset == '20NewsGroup':
        data = fetch_dataset(name="20Newsgroup", remove=("headers", "quotes", "footers"))
    else:
        raise ValueError('Dataset does not exist')
    preprocessor = Pipe(stopwords_list="english", remove_spacy_stopwords=False,
                        token_dict=token_dict, use_spacy_tokenizer=True, min_df=5,
                        max_df=0.80, vocabulary=vocabulary)
    preprocessor.preprocess(None, dataset=data)
    return preprocessor, data


def get_clustering_model(algo='hdbscan', n_clusters=20, metric='euclidean', n_neighbors=10, min_cluster_size=30,
                         random_state=42):
    clustering_model = None
    if algo == 'hdbscan':
        clustering_model = HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, cluster_selection_method='eom',
                                   prediction_data=True)
    elif algo == 'kmeans':
        clustering_model = KMeansClustering(n_clusters=n_clusters, metric=metric)
    elif algo == "gmm":
        clustering_model = GaussianMixture(n_components=n_clusters)
    else:
        raise ValueError("Clustering algorithm not implemented")
    return clustering_model


def get_dim_reduction_model():
    pass


def get_model():
    pass


def get_vocab_embeddings():
    pass
