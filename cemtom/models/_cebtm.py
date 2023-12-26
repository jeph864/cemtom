import joblib

from cemtom.clustering import ClusteringBase
from cemtom.dimreduction import DimensionReductionBase
from cemtom.embedder import EmbedderBase
from cemtom.vectorizers import VectorizerBase

from pathlib import Path
from cemtom.utils import save_utils


class CEBTMBase:
    def __init__(self,
                 language="english",
                 nr_top_words=15,
                 min_topic_size: int = 15,
                 embedding_model: EmbedderBase = None,
                 clustering_model: ClusteringBase = None,
                 dim_reduction_model: DimensionReductionBase = None,
                 vectorizer_model: VectorizerBase = None,
                 topic_representation_model=None,
                 verbose=False,
                 name: str = ''
                 ):
        self.verbose = verbose
        self.topic_representation_model = topic_representation_model
        self.min_topic_size = min_topic_size
        self.vectorizer_model = vectorizer_model
        self.dim_reduction_model = dim_reduction_model
        self.clustering_model = clustering_model
        self.embedding_model = embedding_model
        self.nr_top_words = nr_top_words
        self.language = language
        self.model_name = name
        self.config = dict()
        self.representations = None

    def fit(self):
        pass

    def fit_transform(self):
        raise NotImplementedError

    def transform(self):
        pass

    def get_topics(self):
        pass

    def get_topic(self):
        pass

    def generate_topic_labels(self):
        pass

    def generate_topic_words(self, topic):
        pass

    def extract_embeddings(self):
        pass

    def reduce_dimensionality(self):
        pass

    def cluster_embeddings(self):
        pass

    def vectorize_topics(self):
        pass

    def save(self, path, serialization="pickle", save_embedding_model=True, save_features=False):
        if serialization == "pickle":
            with open(path, 'wb') as f:
                self.vectorizer_model.stop_words_ = None
                if not save_embedding_model:
                    _embedding_model_ = self.embedding_model
                    self.embedding_model = None
                    joblib.dump(self, f)
                    self.embedding_model = _embedding_model_
                else:
                    joblib.dump(self, f)
        elif serialization in ["safetensors", "pytorch"]:
            target = Path(path)
            target.mkdir(exist_ok=True, parents=True)
            # Save
            save_utils.save_embeddings(self, target, serialization)
            save_utils.save_topics(self, target / save_utils.TOPICS_FILE)
            embedding_model = None
            if save_embedding_model and hasattr(self.embedding_model, '_hf_model') and not isinstance(
                    save_embedding_model, str):
                embedding_model = self.embedding_model._hf_model
            save_utils.save_model_config(self, target / save_utils.CONFIG_FILE, embedding_model)

    @classmethod
    def load(cls, path, embedding_model):
        target = Path(path)
        if target.is_file():
            with open(target, 'rb') as r:
                if embedding_model:
                    topic_model = joblib.load(r)
                    topic_model.embedding_model = embedding_model
                else:
                    topic_model = joblib.load(r)
                return topic_model
        if target.is_dir():
            topics, params, tensors, features_tensors, features_cfg = save_utils.load_files(target)
        else:
            raise ValueError("Pass a valid directory")
        topic_model = None

    def evaluate(self, metric):
        pass


def __create_model_(model, topics, params, tensors, features_tensors, features_cfg):
    """Create a CETM Model """
    params["n_gram_range"] = tuple(params["n_gram_range"])
    if features_cfg is not None:
        features_tensors["vectorizer_model"]["params"]["n_gram_range"] = tuple(
            features_tensors["vectorizer_model"]["params"]["n_gram_range"])
    embedding_model = params['embedding_model']
    if params.get("embedding_model") is not None:
        del params['embedding_model']
    base_dim_reduction = DimensionReductionBase()
    base_cluster_model = ClusteringBase()
    topic_model = model(
        embedding_model=embedding_model,
        dim_reduction_model=base_dim_reduction,
        clustering_model=base_cluster_model,
        **params
    )
    topic_model.topic_embeddings = tensors["topic_embeddings"]
    topic_model.topic_representations_ = {int(key): val for key, val in topics["topic_representations"].items()}
    topic_model.topics_ = topics["topics"]
    topic_model.topic_sizes_ = {int(key): val for key, val in topics["topic_sizes"].items()}
    topic_model.topic_labels_ = {int(key): val for key, val in topics["topic_labels"].items()}
    topic_model._outliers = topics["outliers"]
