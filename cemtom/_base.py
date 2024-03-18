import abc
import warnings

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
from torch.utils.data import DataLoader, Dataset as PyDataset

from sklearn.datasets import fetch_20newsgroups

pretrained_models = 'pretrained_models'


class TextDataset(PyDataset):
    def __init__(self, idx2token, bow, embeddings=None, labels=None):
        self.bow = bow
        self.embeddings = embeddings
        self.labels = labels
        self.idx2token = idx2token

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.bow[idx])
        return_item = {'X': X}
        if self.embeddings is not None:
            return_item['X_emb'] = torch.FloatTensor(self.embeddings[idx])
        if self.labels is not None:
            return_item['label'] = self.labels[idx]
        return return_item


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
        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
        train_text, train_labels = train_data['data'], train_data['target']
        test_text, test_labels = test_data['data'], test_data['target']
        val_text, val_labels = test_text, test_labels
        # data = fetch_dataset(name="20Newsgroup", remove=("headers", "quotes", "footers"))
    else:
        raise ValueError('Dataset does not exist')
    preprocessor = Pipe(stopwords_list="english", remove_spacy_stopwords=False,
                        min_chars=3, max_features=2000, min_words=5,
                        token_dict=token_dict, use_spacy_tokenizer=True, min_df=20,
                        max_df=0.80, vocabulary=vocabulary)
    preprocessor.preprocess(train_text + val_text, dataset=None)
    return preprocessor, (train_text, val_text)


def renormalize_bow(train_data, test_data, pipe):
    vocab = pipe.vectorizer.vocabulary_
    for k, special_token in pipe.token_dict.items():
        stripped = special_token.replace("<", "").replace(">", "").replace("/", "").strip()
        vocab.pop(stripped, None)
    train_text, _, _ = pipe.filter_docs_with_vocab(vocab,
                                                   pipe.tokenize(train_data, len(train_data)))
    test_text, _, _ = pipe.filter_docs_with_vocab(vocab, pipe.tokenize(test_data, len(test_data)))
    # tokens = [token for doc in train_text for token in doc.split()]
    # vocabulary = sorted(list(set(tokens)))
    vectorizer = CountVectorizer(vocabulary=list(vocab.keys()))
    X = vectorizer.fit_transform(train_text)
    train_bow = X.toarray()
    test_bow = vectorizer.transform(test_text).toarray()
    vocab = vectorizer.vocabulary_
    return train_bow, test_bow, vocab, train_text, test_text


def get_torch_data(dataset='20NewsGroup', sbert_model=None, max_seq_length=None, batch_size=200, renormalize=False):
    pipe, data = get_data(dataset=dataset)
    train_data, test_data = data
    print(len(train_data), len(test_data))
    train_bow, test_bow = None, None
    vocab = pipe.vectorizer.vocabulary_
    if renormalize:
        train_bow, test_bow, vocab, train_corpus, test_corpus = renormalize_bow(train_data, test_data, pipe)
    else:
        # train_corpus, test_corpus = pipe.data.get_partitioned()
        # train_corpus, test_corpus = pipe.tokenize(train_data), pipe.tokenize(test_data)
        train_bow, train_corpus, train_data = pipe.transform(train_data)
        test_bow, test_corpus, test_data = pipe.transform(test_data)
    text = [doc.split() for doc in train_corpus + test_corpus]
    print(len(train_data), len(test_data))

    idx2token = {v: k for k, v in vocab.items()}

    train_embeddings, val_embeddings = None, None
    if sbert_model is not None:
        model = SentenceTransformer(sbert_model)
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length
            check_max_local_length(max_seq_length, train_data)

        train_embeddings = np.array(model.encode(train_data, show_progress_bar=True, batch_size=batch_size))
        val_embeddings = np.array(model.encode(test_data, show_progress_bar=True, batch_size=batch_size))
    loader = {
        'train': DataLoader(TextDataset(idx2token, train_bow, embeddings=train_embeddings), batch_size, shuffle=True),
        'test': DataLoader(TextDataset(idx2token, test_bow, embeddings=val_embeddings), batch_size, shuffle=False),
        'val': DataLoader(TextDataset(idx2token, test_bow, embeddings=val_embeddings), batch_size, shuffle=False)
    }

    return loader, data, text, vocab


def normalize_bow(train_text, test_text):
    vectorizer = TfidfVectorizer()
    train_bow = vectorizer.fit_transform(train_text)
    train_bow = train_bow.dense().toarray()
    test_bow = vectorizer.transform(test_text).dense().toarray()
    return train_bow, test_bow


def bert_embeddings_from_list(
        texts, sbert_model_to_load, batch_size=200, max_seq_length=None):
    """
    Creates SBERT Embeddings from a list
    """
    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
        check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def check_max_local_length(max_seq_length, texts):
    max_local_length = np.max([len(t.split()) for t in texts])
    if max_local_length > max_seq_length:
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            f"the longest document in your collection has {max_local_length} words, the model instead "
            f"truncates to {max_seq_length} tokens."
        )


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


def run_experiment(name, ):
    pass


def get_dim_reduction_model():
    pass


def get_model():
    pass


def get_vocab_embeddings():
    pass
