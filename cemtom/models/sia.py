import pandas as pd
import numpy as np
from cemtom.embedder import FasttextEmbedder, get_word_embedding_model
from cemtom.clustering import KMeansClustering, HDBSCANClustering
from sklearn.feature_extraction.text import CountVectorizer
from cemtom.dimreduction import PCA as PCAReduction

class Sia:
    def __init__(self,
                 vocab=None,
                 embedding_model_name="fasttext",
                 embedding_model=None,
                 vectorizer=None,
                 nr_dimensions=None,
                 reduction_model=None,
                 nr_topics=10,
                 rerank=None,
                 weighting=None
                 ):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = embedding_model
        self.nr_dimensions = nr_dimensions
        self.nr_topics = nr_topics
        self.vocab = vocab
        self.vocab_embeddings = None
        self.vectorizer = vectorizer
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer()
            pass
        self.reduction_model = reduction_model
        if self.reduction_model is None:
            self.reduction_model = PCAReduction(nr_dimensions)
        if self.vocab is None:
            pass  # raise TypeError("Vocab should not be NoneType")
        self.cluster_model = None
        self._labels = None
        self.rerank = rerank
        self.weighting = weighting
        self.feat_mat = None

    def fit(self, documents, vocab=None, embeddings=None, y=None):
        self.fit_transform(documents=documents, vocab=vocab, embeddings=embeddings, y=y)
        return self

    def fit_transform(self, documents, vocab=None, embeddings=None, y=None):
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
        if embeddings is None and self.embedding_model is None:
            self.embedding_model = get_word_embedding_model(name="fasttext", path="embeds/fasttext/wiki.en.bin")

        elif self.embedding_model is not None:
            print("creating vocabulary embeddings")
            vocab_embeddings = self.embedding_model.embed(self.vocab)
        else:
            vocab_embeddings = embeddings
        self.vocab_embeddings = vocab_embeddings
        print(f"vocab embeddings shape: {self.vocab_embeddings.shape}; documents shape : {len(documents)}")
        if self.reduction_model is not None:
            print("reducing the dimensions")
            vocab_embeddings = self.reduction_model.fit_transform(vocab_embeddings)
        elif self.nr_dimensions is not None:
            print("reducing the dimensions with PCA")
            self.reduction_model = PCAReduction(nr_dims=self.nr_dimensions)
            vocab_embeddings = self.reduction_model.fit_transform(vocab_embeddings)
        # weighting
        weights = None
        if self.weighting is not None:
            if self.weighting == "wgt":
                weights = self.feat_mat.toarray().sum(axis=0)
                print(f"weights shape(before) : {weights[0].shape}")
                print(np.squeeze(weights))
                # scale
        if weights is not None:
            scaled_weights = 1 / (1 + np.exp(weights))
            weights = scaled_weights.reshape(-1)
            print(f"weights shape : {weights.shape}")

        # start clustering
        self.cluster_model = KMeansClustering(n_clusters=self.nr_topics)
        clusters, centers = self.cluster_model.fit_transform(vocab_embeddings, sample_weight=weights)
        #print(f"Finished clustering: {clusters.shape}, centers: {centers.shape}")

        sorted_tops = self._sort_closest_centers(centers, clusters, vocab_embeddings)
        #print(f"Shape of the sorted topk words {sorted_tops.shape}")
        top_k_indices = None
        if self.rerank:
            top_k_indices = self._find_top_k_words(100, sorted_tops)
        else:
            top_k_indices = self._find_top_k_words(10, sorted_tops)
        ##rerank
        self.labels_ = clusters
        self.top_k = top_k_indices
        if self.rerank is not None:
            print("reranking")
            self.top_k = self._rerank(np.array(top_k_indices), documents)
        return self.top_k

    def get_topic_words(self):
        if self.top_k is None:
            raise ValueError("Fit the model first")
        words = []
        for topic in self.top_k:
            words.append(self.vocab[topic])
        return words

    def _rerank(self, topic_word_indices, documents, feat_mat=None, k=20):
        if feat_mat is None:
            feat_mat = self.vectorizer.transform(documents).toarray()
        feat_mat = feat_mat.T
        topk = []
        print(f"feat matrix: {feat_mat.shape}/ topic word indices: {topic_word_indices.shape}")
        for topic_words_idx in topic_word_indices:
            count = feat_mat[topic_words_idx].sum(axis=1)
            count = count.argsort()[-k:][::-1].astype(int)
            topk.append(topic_words_idx[count])
        # print(topk)
        return topk

    # def

    def _get_document_stats(self, weighting=None):
        pass

    def _find_top_k_words(self, k, top_vals):
        topk_words = []
        for top in range(top_vals.shape[0]):
            ind, unique = [], set()
            for i in top_vals[top]:
                word = self.vocab[i]
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
        for topic in range(centers.shape[0]):
            diic = np.where(clusters == topic)[0]
            dist = np.sum((vocab_embedding[diic] - centers[topic]) ** 2, axis=1)
            topk = dist.argsort()[:k]
            # print(words[diic[topk]])
            # print(diic[topk].shape)
            top_idx = np.vstack((top_idx, diic[topk])) if topic > 0 else diic[topk]
        return top_idx


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
