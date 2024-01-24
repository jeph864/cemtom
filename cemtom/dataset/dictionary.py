class Dictionary:
    def __init__(self, texts, vocab=None):
        self.texts = texts
        if self.texts is None:
            raise ValueError("Texts should never be empty")
        self.vocab = vocab
        self.word2idx = {term: index for index, term in enumerate(vocab)}
        self.word2doc_unique, self.word2doc = self.get_word2doc_mapping()

    def get_word2doc_mapping(self, use_vocab=True):
        mapping, mapping_multi = None, None
        if use_vocab:
            if self.vocab is None:
                raise ValueError("Please provide vocabulary")
            mapping = {term: set() for term in self.vocab}
            mapping_multi = {term: [] for term in self.vocab}
            for i, doc in enumerate(self.texts):
                for word in doc:
                    mapping[word].add(i)
                    mapping_multi[word].append(i)
        else:
            vocab = set()
            mapping, mapping_multi = {}, {}
            for i, doc in enumerate(self.texts):
                for word in doc:
                    if word not in vocab:
                        vocab.add(word)
                        mapping[word] = set()
                        mapping[word].add(i)
                        mapping_multi[word] = [i]
                    else:
                        mapping[word].add(i)
                        mapping_multi[word].append(i)
        return mapping, mapping_multi

    def term2doc_matrix(self, w2d):
        pass
