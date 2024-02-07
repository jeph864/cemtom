import os.path
from abc import ABC, abstractmethod
from octis.evaluation_metrics import coherence_metrics, similarity_metrics
from octis.evaluation_metrics.diversity_metrics import (
    TopicDiversity, InvertedRBO, WordEmbeddingsInvertedRBO, WordEmbeddingsInvertedRBOCentroid)
from octis.evaluation_metrics.classification_metrics import (
    F1Score, PrecisionScore, RecallScore, AccuracyScore, ClassificationScore)
from octis.evaluation_metrics import metrics

from cemtom.dataset import Dictionary
from cemtom.utils import download_gdrive, extract

import numpy as np

word2vec_id = '0B7XkCwpI5KDYNlNUTTlSS21pQmM'


class Coherence(coherence_metrics.Coherence):
    def __init__(self, texts=None, topk=10, processes=1):
        super().__init__(texts=texts, topk=topk, processes=processes)
        self.measure = None

    def score(self, model_output, measure='c_npmi'):
        self.measure = measure
        return super().score(model_output)


class WECoherence(metrics.AbstractMetric):
    def __init__(self, word2vec_path=None, binary=False, topk=10):
        super().__init__()
        self.measure = None
        self.word2vec_path = word2vec_path
        self.binary = binary
        self.topk = topk
        if not os.path.exists(self.word2vec_path):
            print(f"Downloading  Word2Vec to {self.word2vec_path}")
            extract(download_gdrive(word2vec_id, cached=True), self.word2vec_path)

    def score(self, model_output, measure='centroid'):
        self.measure = measure
        model = None
        if measure == 'centroid':
            model = coherence_metrics.WECoherenceCentroid(topk=self.topk, word2vec_path=self.word2vec_path,
                                                          binary=self.binary)
        elif measure == 'pairwise':
            model = coherence_metrics.WECoherencePairwise(word2vec_path=self.word2vec_path, binary=self.binary,
                                                          topk=self.topk)
        else:
            raise ValueError('measure does not exist!')
        return model.score(model_output)


class TopicEvaluation:
    def __init__(self, model=None, texts=None, dataset=None, topk=10,
                 ignore_measure=None, word2vec_path=None, word2doc=None,
                 cls_scale=True,
                 cls_average='macro'
                 ):
        self.model = model
        self.dataset = dataset
        self.texts = texts
        self.word2doc=word2doc
        self.topk = topk
        self.cls_scale = cls_scale
        self.cls_average = cls_average
        self.word2vec_path = word2vec_path
        self.ignore_measure = ignore_measure
        # if self.ignore_measure is None:
        self.coherence_metrics = [
            'npmi', 'umass', 'c_v', 'uci', 'we_centroid', 'we_pairwise', 'sia_npmi'
        ]
        self.diversity_metrics = [
            'td', 'irbo', 'we_irbo', 'we_irbo_centroid'
        ]
        self.classification_metrics = [
            'f1', 'recall', 'precision', 'accuracy'
        ]
        self.scores = {}

    def classification_single(self, measure='f1'):
        score = None
        if 'topic-document-matrix' not in self.model or (
                'topic-document-matrix' in self.model and self.model['topic-document-matrix'] is None):
            return
        if measure == 'f1':
            score = F1Score(dataset=self.dataset, scale=self.cls_scale, average=self.cls_average)
        elif measure == 'recall':
            score = RecallScore(dataset=self.dataset, scale=self.cls_scale, average=self.cls_average)
        elif measure == 'precision':
            score = PrecisionScore(dataset=self.dataset, scale=self.cls_scale, average=self.cls_average)
        elif measure == 'accuracy':
            score = AccuracyScore(dataset=self.dataset, scale=self.cls_scale, average=self.cls_average)
        else:
            raise ValueError("Metric does not exist")
        return score.score(self.model)

    def classification_score(self):
        return {measure: self.classification_single(measure) for measure in self.classification_metrics}

    def coherence_score(self, model_output=None, remove=None):
        coherence_model = Coherence(self.texts, topk=self.topk)
        we_coherence_model = WECoherence(word2vec_path=self.word2vec_path, topk=self.topk, binary=True)
        scores = {}
        measures = self.coherence_metrics
        if model_output is None:
            model_output = self.model
        if remove is not None:
            if isinstance(remove, list) or isinstance(remove, tuple):
                measures = [measure for measure in self.coherence_metrics if measure not in remove]
        if 'npmi' in measures:
            scores['npmi'] = coherence_model.score(model_output, 'c_npmi')
        if 'umass' in measures:
            scores['umass'] = coherence_model.score(model_output, 'u_mass')
        if 'uci' in measures:
            scores['uci'] = coherence_model.score(model_output, 'c_uci')
        if 'c_v' in measures:
            scores['c_v'] = coherence_model.score(model_output, 'c_v')
        if 'centroid' in measures:
            scores['we_coherence_centroid'] = we_coherence_model.score(model_output, 'centroid')
        if 'pairwise' in measures:
            scores['we_coherence_pairwise'] = we_coherence_model.score(model_output, 'pairwise')
        if 'sia_npmi' in measures:
            scores['sia_npmi'] = self.sia_npmi()
        return scores

    def sia_npmi(self, word_doc_counts=None, nfiles=None):
        if nfiles is None:
            nfiles = len(self.texts)
        if word_doc_counts is None:
            word_doc_counts = self.word2doc
        eps = 10 ** (-12)
        topic_words = self.model['topics']
        ntopics = len(topic_words)

        all_topics = []
        for k in range(ntopics):
            word_pair_counts = 0
            topic_score = []

            ntopw = len(topic_words[k])

            for i in range(ntopw - 1):
                for j in range(i + 1, ntopw):
                    w1 = topic_words[k][i]
                    w2 = topic_words[k][j]
                    w1w2_dc = len(word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set()))
                    w1_dc = len(word_doc_counts.get(w1, set()))
                    w2_dc = len(word_doc_counts.get(w2, set()))
                    pmi_w1w2 = np.log((w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps)
                    npmi_w1w2 = pmi_w1w2 / (- np.log((w1w2_dc) / nfiles + eps))
                    topic_score.append(npmi_w1w2)

            all_topics.append(np.mean(topic_score))

        for k in range(ntopics):
            pass
            # print(np.around(all_topics[k], 5), " ".join(topic_words[k]))

        avg_score = np.around(np.mean(all_topics), 5)
        # print(f"\nAverage NPMI for {ntopics} topics: {avg_score}")

        return avg_score

    def diversity_score(self, model_output=None, remove=None):
        if model_output is None:
            model_output = self.model
        scores = {
            'td': TopicDiversity().score(model_output),
            'irbo': InvertedRBO().score(model_output),
            # 'we_irbo': WordEmbeddingsInvertedRBO(word2vec_path=self.word2vec_path, binary=True).score(
            # model_output), 'we_irbo_centroid': WordEmbeddingsInvertedRBOCentroid(word2vec_path=self.word2vec_path,
            # binary=True).score(model_output)
        }
        return scores

    def evaluate(self):
        return {
            'coherence': self.coherence_score(),
            'diversity': self.diversity_score(),
            'classification': self.classification_score()
        }
