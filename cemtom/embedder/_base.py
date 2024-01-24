# Import necessary libraries
from sentence_transformers import SentenceTransformer
import cohere
from flair.embeddings import TransformerDocumentEmbeddings
#import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


# Custom Exception for Unfitted Pipeline
class PipelineNotFittedError(Exception):
    pass


# Global function to check if a pipeline is fitted
def is_pipeline_fitted(pipeline):
    try:
        check_is_fitted(pipeline)
        return True
    except:
        return False


# Base Class for Embedders
class EmbedderBase:
    def __init__(self, **params):
        self.params = params

    def embed(self, text):
        raise NotImplementedError("This method should be implemented by subclasses.")


class SentenceTransformersEmbedder(EmbedderBase):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super().__init__(model_name=model_name)
        self.model = SentenceTransformer(model_name)

    def embed(self, documents):
        return self.model.encode(documents, show_progress_bar=False)


# Flair Embedder
class FlairEmbedder(EmbedderBase):
    def __init__(self, model='bert-base-uncased'):
        super().__init__(model=model)
        self.model = TransformerDocumentEmbeddings(model)

    def embed(self, docs):
        from flair.data import Sentence
        sentence = Sentence(docs)
        self.model.embed(sentence)
        return sentence.get_embedding()


# Cohere Embedder with Batch Size Handling
class CohereEmbedder(EmbedderBase):
    def __init__(self, api_key, model='large', batch_size=None):
        super().__init__(api_key=api_key, model=model, batch_size=batch_size)
        self.model = cohere.Client(api_key)

    def embed(self, texts):
        if self.params['batch_size']:
            return [
                self.model.embed(texts=texts[i:i + self.params['batch_size']], model=self.params['model']).embeddings
                for i in range(0, len(texts), self.params['batch_size'])]
        else:
            return self.model.embed(texts=texts, model=self.params['model']).embeddings


# Sklearn Embedder with Pipeline Support
class SklearnEmbedder(EmbedderBase):
    def __init__(self, pipeline=None):
        super().__init__(pipeline=pipeline)
        self.pipeline = pipeline or TfidfVectorizer()

    def embed(self, documents):
        if not is_pipeline_fitted(self.pipeline):
            if isinstance(self.pipeline, Pipeline):
                self.pipeline.fit_transform(documents)
            else:
                raise PipelineNotFittedError("Pipeline must be fitted before embedding.")
        return self.pipeline.fit_transform(documents)


# Universal Sentence Encoder (USE) Embedder
class USEEmbedder(EmbedderBase):
    def __init__(self, model_url='https://tfhub.dev/google/universal-sentence-encoder/4'):
        super().__init__(model_url=model_url)
        self.model = None#hub.load(model_url)

    def embed(self, documents):
        return self.model(documents)
