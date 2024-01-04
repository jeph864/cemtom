import fasttext.util
import fasttext
import os

import numpy as np


class BaseWordEmbedder:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def embed(self, words):
        raise NotImplemented("Subclass should implement the method")


class FasttextEmbedder(BaseWordEmbedder):
    def __init__(self, embedding_model=None, path=None):
        super().__init__()
        self.embedding_model = embedding_model
        self.model_path = path
        self.model = None
        if self.model_path is not None and os.path.exists(self.model_path):
            self.model = fasttext.load_model(self.model_path)
        elif self.embedding_model is not None:
            self.model = self.embedding_model
        else:
            raise ValueError("No Model or path given")

    def embed(self, words):
        word_embeddings = []
        for word in words:
            word_embeddings.append(self.model.get_word_vector(word))
        return np.array(word_embeddings)


def get_word_embedding_model(name=None, model=None, path=None):
    if name == "fasttext":
        return FasttextEmbedder(embedding_model=model, path=path)
    else:
        return BaseWordEmbedder()