import json
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

LABEL_SEPARATOR = ','
def load_json(filename):
    file = Path(filename)
    data = dict()
    if file.is_file():
        with open(filename, 'r') as infile:
            data = json.load(infile)
    return data


def save_json(filename, data):
    if data is not None:
        with open(filename, 'w') as out:
            json.dump(data, out)
    else:
        raise Exception("error in saving file")


def save_iterable_text(filename, data):
    # file = Path(filename)
    with open(filename, 'w') as outfile:
        for i in data:
            if isinstance(i, list):
                i = ",".join(i)
            outfile.write(str(i) + "\n")


def load_iterable_text(filename):
    data = []
    file = Path(filename)
    if file.is_file():
        with open(filename, 'r') as infile:
            for line in infile:
                data.append(line.strip())
    return data if len(data) > 0 else None


def load_dataset(path, name='custom', multilabel=False):
    vocabulary = load_iterable_text(path + '/vocab.txt')
    metadata = load_json(path + '/metadata.json')
    indices = load_iterable_text(path + '/indices.txt')
    documents = load_iterable_text(path + '/documents.txt')
    labels = load_iterable_text(path + '/labels.txt')
    if multilabel:
        labels = [label.split(',') for label in labels]
    return Dataset(docs=documents, labels=labels, vocabulary=vocabulary, indices=indices, metadata=metadata)


class Dataset:
    def __init__(self,
                 docs=None,
                 vocabulary=None,
                 labels=None,
                 metadata=None,
                 indices=None,
                 preprocessed=False,
                 name='custom'
                 ):
        self.__df = None
        self.__indices = indices
        self.__labels = labels
        self.__vocabulary = vocabulary
        self.__metadata = metadata if metadata else dict()
        self.__corpus = docs
        self.__path = None
        self.__cache = False
        self.name = name
        self.preprocessed = preprocessed

    def _clean(self):
        if self.__corpus is not None:
            pass

    def get_corpus(self):
        return self.__corpus

    def get_labels(self):
        return self.__labels

    def get_vocabulary(self):
        return self.__vocabulary

    def get_info(self):
        return self.__metadata

    def __len__(self):
        return 0 if self.__corpus is None else len(self.__corpus)

    def __getitem__(self, idx):
        item = None
        if (self.__labels is not None and len(self.__labels) > 0) and self.__corpus is not None:
            item = (self.__corpus[idx], self.__labels[idx])
        elif self.__corpus is not None:
            item = (self.__corpus[idx], '')
        else:
            raise IndexError("Index out of bound")

        return item

    def get_indices(self):
        return self.__indices

    def tokenize(self):
        return [d.split() for d in self.__corpus]

    def split(self, test_size=0.15, validation_size=0, random_state=None, stratify=None, shuffle=False):
        validation_indices = []
        if self.__labels is not None:
            train_indices, test_indices = train_test_split(
                range(len(self.__corpus)), test_size=test_size, random_state=random_state, stratify=stratify,
                shuffle=shuffle)
            if validation_size > 0:
                train_indices, validation_indices = train_test_split(
                    train_indices, test_size=validation_size, random_state=random_state, shuffle=shuffle,
                    stratify=[self.__labels[i] for i in train_indices])
            # return train_indices, validation_indices, test_indices
        else:
            train_indices, test_indices = train_test_split(
                range(len(self.__corpus)), test_size=test_size, random_state=random_state, shuffle=shuffle)
            if validation_size > 0:
                train_indices, validation_indices = train_test_split(
                    train_indices, test_size=validation_size, random_state=random_state, shuffle=shuffle)
        new_corpus, new_labels = [], []

        new_corpus_indices = train_indices
        new_corpus_indices.extend(validation_indices)
        new_corpus_indices.extend(test_indices)
        for i in new_corpus_indices:
            new_corpus.append(self.__corpus[i])
        if self.__labels is not None:
            for i in new_corpus_indices:
                new_labels.append(self.__labels[i])
        if validation_size != 0:
            self.__metadata["validation_idx"] = len(train_indices)
        self.__metadata["test_idx"] = len(train_indices) + len(validation_indices)
        self.__corpus = new_corpus
        self.__labels = new_labels
        self.__indices = new_corpus_indices

    def get_partitioned(self, return_validation=False, corpus=True):
        if self.__corpus is None:
            raise CorpusNotFoundError()
        if "test_idx" not in self.__metadata:
            return [self.__corpus, None]
        train_corpus, test_corpus = [], []
        if return_validation:
            if "validation_idx" in self.__metadata:
                validation_start = self.__metadata["validation_idx"]
                if validation_start != 0:
                    validation_corpus = []
                    test_idx = self.__metadata["test_idx"] if "test_idx" in self.__metadata else 0
                    for i in range(test_idx):
                        train_corpus.append(self.__corpus[i])
                    for i in range(validation_start, test_idx):
                        train_corpus.append(self.__corpus[i])
                    for i in range(test_idx, len(self.__corpus)):
                        test_corpus.append(self.__corpus[i])
                    return train_corpus, validation_corpus, test_corpus

        else:
            test_idx = self.__metadata["test_idx"] if "test_idx" in self.__metadata else 0

            for i in range(test_idx):
                train_corpus.append(self.__corpus[i])
            for i in range(test_idx, len(self.__corpus)):
                test_corpus.append(self.__corpus[i])
            return train_corpus, test_corpus

    def save(self, path, remove_partitions=False, remove_empty_docs=True, dummy_label='label', mallet=False):

        metadata_path = '/metadata.json'
        labels_path = '/labels.txt'
        vocab_path = '/vocab.txt'
        corpus_path = '/corpus.tsv'

        Path(path).mkdir(parents=True, exist_ok=True)
        parts = self.get_partitioned()
        train, test = [], []
        if len(parts) == 2:
            train, test = parts
        # parts = (train, test)
        docs = []
        partition = []
        for i, part in enumerate(parts):
            if i == 0:
                corpus_type = 'train'
            if i == 1 and len(parts) == 3:
                corpus_type = 'val'
            elif i == 1 and len(parts) != 3:
                corpus_type = 'test'
            docs.extend(part)
            partition.extend([corpus_type] * len(part))
        data = dict()
        doc_indices = []
        if self.__indices is not None:
            doc_indices.extend(self.__indices)
        else:
            doc_indices = np.arange(len(docs))
        data['indices'] = doc_indices
        if not isinstance(self.__labels, list) or len(self.__labels) == 0:
            data['labels'] = [dummy_label] * len(docs)
        else:
            data['labels'] = self.__labels
            if isinstance(self.__labels[0], list):
                data['labels'] = data['labels'].apply(lambda x: ','.join(x))

        data['documents'] = docs
        if not remove_partitions:
            data['partition'] = partition
        elif not mallet:
            data['partition'] = ['train'] * len(docs)
        df = pd.DataFrame(data)
        if remove_empty_docs:
            df = df[~df['documents'].isnull()]
        df.to_csv(path + corpus_path, '\t', index=False, header=False)
        save_json(path + metadata_path, self.__metadata)
        if self.__labels is not None:
            save_iterable_text(path + labels_path, self.__labels)
        if self.__vocabulary is not None:
            save_iterable_text(path + vocab_path, self.__vocabulary)
        if self.__indices is not None:
            save_iterable_text(path + '/indices.txt', self.__indices)
        if self.__corpus:
            save_iterable_text(path + '/documents.txt', docs)
        self.__path = path

    def load_from_tsv(self, path):
        self.__path = path
        self.__vocabulary = load_iterable_text(self.__path + '/vocabulary.txt')
        self.__metadata = load_json(self.__path + '/metadata.json')
        tmp_df = pd.read_csv(self.__path + "/corpus.tsv", sep='\t', header=None)
        corpus = pd.concat([tmp_df[tmp_df[3] == 'train'], tmp_df[tmp_df[3] == 'val'], tmp_df[tmp_df[3] == 'test']])
        self.__corpus = [d for d in corpus[2].tolist()]
        self.__labels = [d for d in corpus[1].tolist()]
        validation_start = 0
        test_start = len(corpus[corpus[3] == 'val']) + len(corpus[corpus[3] == 'train'])
        if len(corpus[corpus[3] == 'val']) > 0:
            validation_start = len(corpus[corpus[3] == 'train'])
            if validation_start != 0 and not validation_start <= test_start:
                del self.__metadata['validation_idx']
                del self.__metadata['test_idx']

        self.__metadata['validation_idx'] = validation_start
        self.__metadata['test_idx'] = test_start
        corpus.columns = ['indices', 'labels', 'documents', 'partition']
        self.__df = corpus
        return self.__df


class CorpusErrorException(Exception):
    def __init__(self, message="Corpus Error. "):
        super().__init__(message)


class CorpusNotFoundError(CorpusErrorException):
    def __init__(self, message="Corpus not found."):
        super().__init__(message)


class CorpusInvalidStructureException(CorpusErrorException):
    def __init__(self, message="Invalid Structure"):
        super().__init__(message)


def download_dataset(dataset_name):
    pass
