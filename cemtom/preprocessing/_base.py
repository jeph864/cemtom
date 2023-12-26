import string
from pathlib import Path

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from collections import Counter
from ..dataset.dataset import Dataset, CorpusNotFoundError

spacy_model_mapping = {
    'chinese': 'zh_core_web_sm', 'danish': 'nl_core_news_sm',
    'dutch': 'nl_core_news_sm', 'english': 'en_core_web_sm',
    'french': 'fr_core_news_sm', 'german': 'de_core_news_sm',
    'greek': 'el_core_news_sm', 'italian': 'it_core_news_sm',
    'japanese': 'ja_core_news_sm', 'lithuanian': 'lt_core_news_sm',
    'norwegian': 'nb_core_news_sm', 'polish': 'pl_core_news_sm',
    'portuguese': 'pt_core_news_sm', 'romanian': 'ro_core_news_sm',
    'russian': 'ru_core_news_sm', 'spanish': 'es_core_news_sm'}


class Preprocessor:
    def __init__(self,
                 lowercase=True, lemmatize=False,
                 remove_punctuation=True, remove_numbers=True, language="english",
                 min_df=0.0, max_df=1.0, min_words=0, punctuation=string.punctuation,
                 min_chars=1, stopwords_list=None,
                 num_process=None, remove_spacy_stopwords=True, preprocessed=False,
                 should_split = True
                 ):
        self.vectorizer = None
        self.min_chars = min_chars
        self.vocabulary = None
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.stopwords = []
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_df = min_df
        self.max_df = max_df
        self.min_words = min_words
        self.punctuation = punctuation
        self.remove_spacy_stopwords = remove_spacy_stopwords
        self.language = language
        self.max_features = None
        self.num_processes = num_process
        self.should_split = True

        self.preprocessed = preprocessed

        if self.lemmatize:
            lang = spacy_model_mapping[self.language]
            try:
                self.spacy_model = spacy.load(lang)
            except IOError:
                raise IOError("Can't find model " + lang + ". Check the data directory or download it using the "
                                                           "following command:\npython -m spacy download " + lang)
        stopwords = []
        if stopwords_list is None:
            self.remove_spacy_stopwords = False
        else:
            # if custom list is specified, then we do not use spacy stopwords
            if type(stopwords_list) == list:
                stopwords = set(stopwords_list)
                self.remove_spacy_stopwords = False
            elif self.remove_spacy_stopwords:
                assert stopwords_list == language
            else:
                # if remove_stopwords_spacy is false, then use MALLET English stopwords
                if 'english' in stopwords_list:
                    stop_word_path = Path(__file__).parent.joinpath('stopwords', 'english.txt')
                    with open(stop_word_path) as fr:
                        stopwords = [line.strip() for line in fr.readlines()]
                        assert stopwords_list == language
        self.stopwords = stopwords
        print(stopwords)

    def split(self, docs_docs, docs_labels=None, docs_idx=None):
        if docs_labels is None:
            docs_labels = []
        metadata = {
            'validation_idx': 0,
            'test_idx': 0
        }
        part_labels, part_corpus, doc_indexes = docs_labels, docs_docs, docs_idx
        if len(docs_labels) > 0:
            train, test, y_train, y_test = train_test_split(
                range(len(docs_docs)), docs_labels, test_size=0.15, random_state=1,
                shuffle=True)

            train, validation = train_test_split(train, test_size=3 / 17, random_state=1,
                                                 shuffle=True)  # stratify=y_train)

            part_labels = [docs_labels[doc] for doc in train + validation + test]
            part_corpus = [docs_docs[doc] for doc in train + validation + test]
            doc_indexes = [docs_idx[doc] for doc in train + validation + test]
            metadata['validation_idx'] = len(train)
            metadata['test_idx'] = len(train) + len(validation)
        else:
            train, test = train_test_split(range(len(docs_docs)), test_size=0.15, random_state=1)
            train, validation = train_test_split(train, test_size=3 / 17, random_state=1)

            part_corpus = [docs_docs[doc] for doc in train + validation + test]
            doc_indexes = [docs_idx[doc] for doc in train + validation + test]
            metadata['validation_idx'] = len(train)
            metadata['test_idx'] = len(train) + len(validation)

        return part_corpus, part_labels, doc_indexes, metadata

    def simple_steps(self, doc):
        new_d = doc
        new_d = new_d.replace('\n', '')
        new_d = new_d.replace('\t', '')
        if self.lowercase:
            new_d = new_d.lower()
        if self.lemmatize:
            if self.remove_spacy_stopwords:
                new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d) if not token.is_stop])
            elif self.stopwords:
                new_d = ' '.join(
                    [token.lemma_ for token in self.spacy_model(new_d) if token.lemma_ not in set(self.stopwords)])
            else:
                new_d = ' '.join([token.lemma_ for token in self.spacy_model(new_d)])

        if self.remove_punctuation:
            new_d = new_d.translate(str.maketrans(self.punctuation, ' ' * len(self.punctuation)))
        if self.remove_numbers:
            new_d = new_d.translate(str.maketrans("0123456789", ' ' * len("0123456789")))
        new_d = " ".join(new_d.split())
        return new_d

    def filter(self, docs):
        print(f"Filtering {len(docs)}")
        print(docs[0])
        if self.vocabulary is not None:
            vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, vocabulary=self.vocabulary,
                                         token_pattern=r"(?u)\b\w{" + str(self.min_chars) + ",}\b",
                                         lowercase=self.lowercase, stop_words=self.stopwords)

        elif self.max_features is not None:
            vectorizer = TfidfVectorizer(lowercase=self.lowercase, max_features=self.max_features,
                                         stop_words=self.stopwords,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b")

        else:
            vectorizer = TfidfVectorizer(max_df=self.max_df, min_df=self.min_df, lowercase=self.lowercase,
                                         token_pattern=r"(?u)\b[\w|\-]{" + str(self.min_chars) + r",}\b",
                                         stop_words=self.stopwords)

        self.vectorizer = vectorizer

        vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names_out()
        return vocabulary

    def preprocess(self, docs_path, labels_path=None, num_processes=None, dataset=None, split = False):
        self.should_split = split
        docs, labels = [], []
        doc_map_list = None
        docs_idx, docs_labels, docs_docs = [], [], []
        if dataset is not None:
            docs = dataset.get_corpus()
            docs_docs = docs
            labels = dataset.get_labels()
            docs_idx = dataset.get_indices()
            docs_labels = labels
        else:
            if docs_path is None:
                raise CorpusNotFoundError()
            else:
                with open(docs_path, 'r') as infile:
                    docs = [line.strip() for line in infile.readlines()]
        if num_processes is not None:
            chunksize = max(1, len(docs) // (num_processes * 20))
            docs_list = process_map(self.simple_steps, docs, chunksize=chunksize, max_workers=num_processes)
            docs = docs_list
        else:
            docs = list(map(self.simple_steps, tqdm(docs)))

        vocabulary = self.filter(docs)
        print(f"vocab created {len(vocabulary)}")
        if dataset is None and labels_path is not None:
            with open(labels_path, 'r') as lines:
                labels = [line.strip() for line in lines.readlines()]
        if len(labels) == 0:
            labels = None
        vocab = set(vocabulary)
        docs_docs, docs_labels, docs_idx = self.filter_docs_with_vocab(vocab, docs=docs, labels=labels)
        print("Filtering done")
        metadata = {"total_documents": len(docs), "vocabulary_length": len(vocabulary)}
        part_labels = None
        if self.should_split:
            part_corpus, part_labels, doc_indexes, part_metadata = self.split(docs_docs, docs_labels, docs_idx)
            metadata['validation_idx'] = part_metadata['validation_idx']
            metadata['test_idx'] = part_metadata['test_idx']
            return Dataset(docs=part_corpus, vocabulary=vocabulary, metadata=metadata, labels=part_labels,
                           indices=doc_indexes)
        else:
            return Dataset(docs=docs_docs, vocabulary=vocabulary, metadata=metadata, labels=docs_labels,
                           indices=docs_idx)

    def filter_docs_with_vocab(self, vocab, docs=None, labels=None):
        if docs is None:
            raise CorpusNotFoundError("Documents not given")
        docs_docs, docs_labels, docs_idx = [], [], []
        if labels is not None:
            for i, doc, label in zip(range(len(docs)), docs, labels):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_words:
                    docs_docs.append(" ".join(new_doc))
                    docs_labels.append(label)
                    docs_idx.append(i)
            labels_to_remove = set([k for k, v in dict(
                Counter(docs_labels)).items() if v <= 3])
            if len(labels_to_remove) > 0:
                docs = docs_docs
                labels = docs_labels
                docs_idx, docs_labels, docs_docs = [], [], []
                for i, doc, label in zip(range(len(docs)), docs, labels):
                    if label not in labels_to_remove:
                        docs_docs.append(doc)
                        docs_labels.append(label)
                        docs_idx.append(i)
        else:
            for i, doc in enumerate(docs):
                new_doc = [w for w in doc.split() if w in vocab]
                if len(new_doc) > self.min_words:
                    docs_docs.append(new_doc)
                    docs_idx.append(i)
        return docs_docs, docs_labels, docs_idx
