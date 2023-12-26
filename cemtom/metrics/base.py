from octis.evaluation_metrics import coherence_metrics, similarity_metrics, diversity_metrics


class TopicEvaluation:
    def __init__(self, vocab, text):
        vocab = {v: k for k, v in vocab.items()}
