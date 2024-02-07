import pandas as pd
import numpy as np

from cemtom.dataset import Dictionary
from cemtom.embedder import FasttextEmbedder, get_word_embedding_model
from cemtom.clustering import KMeansClustering, HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from cemtom.dimreduction import PCA as PCAReduction
from cemtom.models import CEBTMBase


class Sia(CEBTMBase):
    def __init__(self, vocab=None, embedding_model_name="fasttext", embedding_model=None, vectorizer=None,min_topic_size=10,
                 cluster_model = None,
                 nr_dimensions=None, reduction_model=None, nr_topics=10, rerank=None, weighting=None, word2idx=None):
        super().__init__(embedding_model=embedding_model, dim_reduction_model=reduction_model, name="sia")

        self.embedding_model_name = embedding_model_name
        self.embedding_model = embedding_model
        self.nr_dimensions = nr_dimensions
        self.nr_topics = nr_topics
        self.vocab = vocab
        self.vocab_embeddings = None
        self.word2idx = word2idx
        self.use_ext_mapping = True
        if self.word2idx is not None:
            self.use_ext_mapping = True
        self.vectorizer = vectorizer
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer()
            pass
        self.dim_reduction_model = reduction_model
        if self.dim_reduction_model is None:
            self.dim_reduction_model = PCAReduction(nr_dimensions)
        if self.vocab is None:
            pass  # raise TypeError("Vocab should not be NoneType")
        self.clustering_model = cluster_model
        self.labels_ = None
        self.rerank = rerank
        self.weighting = weighting
        self.feat_mat = None
        self.top_k = None
        self.scale = None
        self.topic_words = None

    def fit(self, documents, vocab=None, embeddings=None, y=None):
        self.fit_transform(documents=documents, vocab=vocab, embeddings=embeddings, y=y)
        return self

    def fit_transform(self, documents, word2doc=None, vocab=None, embeddings=None, y=None):
        if vocab is not None:
            self.vocab = vocab
        elif self.vocab is None:
            if self.vectorizer is None:
                raise ValueError("Please provide the vectorozer")
            else:
                # self.vocab = self.vectorizer.get_feature_names_out()
                # if len(self.vocab) > 0:
                #    self.vocab = self.vectorizer.get_feature_names_out()
                # else:
                self.feat_mat = self.vectorizer.fit_transform(documents)
                self.vocab = self.vectorizer.get_feature_names_out()
        else:
            self.feat_mat = self.vectorizer.fit_transform(documents)
        if documents is not None:
            # check_documents_type(documents)
            # check_embeddings_shape(embeddings, self.vocab)
            pass
        if self.vocab is None:
            pass

        docs_ids = range(len(documents))
        docs = pd.DataFrame({"Document": documents,
                             "ID": docs_ids,
                             "Topic": None})
        vocab_embeddings = None
        if embeddings is None:
            if self.embedding_model is None:
                self.embedding_model = get_word_embedding_model(name="fasttext", path="embeds/fasttext/wiki.en.bin")
            print("creating vocabulary embeddings")
            self.vocab_embeddings = self.embedding_model.embed(self.vocab)
        elif embeddings is not None:
            self.vocab_embeddings = embeddings
        else:
            raise ValueError("No Embeddings or embeddings model provided")
        print(f"vocab embeddings shape: {self.vocab_embeddings.shape}; documents shape : {len(documents)}")
        if self.dim_reduction_model is not None:
            print("reducing the dimensions")
            vocab_embeddings = self.dim_reduction_model.fit_transform(self.vocab_embeddings)
        elif self.nr_dimensions is not None:
            print("reducing the dimensions with PCA")
            self.dim_reduction_model = PCAReduction(nr_dims=self.nr_dimensions)
            vocab_embeddings = self.dim_reduction_model.fit_transform(self.vocab_embeddings)
        # weighting
        weights = None
        if self.weighting is not None:
            if self.weighting == "wgt":
                if word2doc is not None:
                    weights = np.array([len(word2doc[word]) for word in self.vocab])
                else:
                    weights = self.feat_mat.toarray().sum(axis=0)
                    print(f"weights shape(before) : {weights[0].shape}")
                # print(np.squeeze(weights))
                # scale
        if weights is not None and self.scale is not None:
            scaled_weights = 1 / (1 + np.exp(weights))
            weights = scaled_weights.reshape(-1)
            print(f"weights shape : {weights.shape}")

        # start clustering
        self.labels_, self.top_k = self._cluster_embeddings(vocab_embeddings, weights, word2doc)
        ranked_topk_words = None
        ranked_topk_idx = None
        if self.rerank is not None:
            if self.rerank == "tf":
                ranked_topk_idx, ranked_topk_words = self._rerank_freq(word2doc)
            else:
                ranked_topk_idx, ranked_topk_words = self._rerank(np.array(self.top_k), None)
        self.top_k = ranked_topk_idx
        self.topic_words = ranked_topk_words
        return self.top_k, ranked_topk_words

    def _cluster_embeddings(self, vocab_embeddings, weights, word2doc=None):
        if self.clustering_model is None:
            self.clustering_model = KMeansClustering(n_clusters=self.nr_topics)
            print("Using KMeans")
        topk_word_idx = self.clustering_model.fit_transform(vocab_embeddings, sample_weight=weights, k=200)
        print(f"sorted tops shape: {topk_word_idx.shape}")
        # print(f"Shape of the sorted topk words {sorted_tops.shape}")
        top_k_indices = None
        if self.rerank:
            top_k_indices = self._find_top_k_words(100, topk_word_idx)
        else:
            top_k_indices = self._find_top_k_words(10, topk_word_idx)
        return self.clustering_model.m_clusters, top_k_indices
    def _weight_words(self):
        pass

    def get_topic_words(self, force=False):
        if self.topic_words is not None and not force:
            return self.topic_words
        if self.top_k is None:
            raise ValueError("Fit the model first")
        words = []
        vocab = self.vocab
        if self.use_ext_mapping:
            vocab = np.array(list(self.word2idx.keys()))
        for topic in self.top_k:
            words.append(vocab[topic])
        return words

    def _rerank(self, topic_word_indices, documents, feat_mat=None, k=20):
        if feat_mat is None:
            feat_mat = self.vectorizer.transform(documents).toarray()
        feat_mat = feat_mat.T
        topk = []
        topic_words = self.get_topic_words()
        print(f"feat matrix: {feat_mat.shape}/ topic word indices: {topic_word_indices.shape}")
        for topic_words_idx, topic in zip(topic_word_indices, topic_words):
            count = feat_mat[topic_words_idx].sum(axis=1)
            count = count.argsort()[-k:][::-1].astype(int)
            topk.append(topic[topic_words_idx[count]])
        # print(topk)
        return topk

    def _rerank_freq(self, word2doc, k=20):
        topk_indices, topk_words = [], []
        topic_words = self.get_topic_words()
        for topic in topic_words:
            count = np.array([len(word2doc[word]) for word in topic])
            topk_idx = count.argsort()[-k:][::-1].astype(int)
            topk_indices.append(topk_idx)
            topk_words.append(topic[topk_idx])
        #self.topic_words = topk_words
        return topk_indices, topk_words

    def _rank_tf_idf(self):
        topic_words = self.get_topic_words()
        raise NotImplementedError("")

    # def

    def _get_document_stats(self, weighting=None):
        pass

    def _find_top_k_words(self, k, top_vals):
        topk_words = []
        vocab = self.vocab
        if self.use_ext_mapping:
            vocab = np.array(list(self.word2idx.keys()))
        for top in range(top_vals.shape[0]):
            ind, unique = [], set()
            for i in top_vals[top]:
                word = vocab[i]
                # print(word)
                if word not in unique:
                    ind.append(i)
                    unique.add(word)
                    if len(unique) == k:
                        break
            topk_words.append(ind)
        return topk_words

    def _sort_closest_centers(self, centers, clusters, vocab_embedding, k=20):
        top_idx = []
        unique, counts = np.unique(clusters, return_counts=True)
        k = np.min((k, np.min(counts)))
        for topic in range(centers.shape[0]):
            diic = np.where(clusters == topic)[0]
            dist = np.sum((vocab_embedding[diic] - centers[topic]) ** 2, axis=1)
            topk = dist.argsort()[:k]
            top_idx = np.vstack((top_idx, diic[topk])) if topic > 0 else diic[topk]
        return top_idx

    def __sort_dist2center(self, centers, clusters, vocab_embeddings, k):
        unique, counts = np.unique(clusters, return_counts=True)
        k = np.min((k, np.min(counts)))
        top_idx = []
        #print({top : cout for top, cout in zip(unique, counts)})
        #print((centers.shape, clusters.shape, vocab_embeddings.shape, (min_k, k)))
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


def sia_dataset_preprocess(docs):
    vocab = set()
    mapping = {}
    for i, doc in enumerate(docs):
        words = doc.split()
        for word in words:
            if word not in vocab:
                vocab.add(word)
                mapping[word] = set()
                mapping[word].add(i)
            else:
                mapping[word].add(i)
    return mapping
