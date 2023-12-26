import codecs
import pickle
import shutil
from sklearn.datasets import load_files
import os
import logging
from contextlib import suppress
import re
import tarfile
import numpy as np
from sklearn.utils import check_random_state

import joblib
from sklearn.datasets._base import load_descr

from .dataset import Dataset

logger = logging.Logger(__name__)

from . import _fetch, _pkl_filepath, get_data_home

CACHE_NAME = "20news-bydate.pkz"
TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"

NEWSGROUP_REMOTE = {
    "url": "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz",
    "filename": "20news-bydate.tar.gz"
}


def _download_dataset(target_dir, cache_path):
    """Download the 20 newsgroups data and stored it as a zipped pickle."""
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)

    os.makedirs(target_dir, exist_ok=True)

    print("Downloading dataset from %s (14 MB)", NEWSGROUP_REMOTE['url'])
    archive_path = _fetch(filename=NEWSGROUP_REMOTE['filename'], url=NEWSGROUP_REMOTE['url'], dirname=target_dir)

    logger.debug("Decompressing %s", archive_path)
    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)

    with suppress(FileNotFoundError):
        os.remove(archive_path)

    # Store a zipped pickle
    cache = dict(
        train=load_files(train_path, encoding="latin1"),
        test=load_files(test_path, encoding="latin1"),
    )
    compressed_content = codecs.encode(pickle.dumps(cache), "zlib_codec")
    with open(cache_path, "wb") as f:
        f.write(compressed_content)

    shutil.rmtree(target_dir)
    return cache


def strip_newsgroup_header(text):
    _before, _blankline, after = text.partition("\n\n")
    return after


_QUOTE_RE = re.compile(
    r"(writes in|writes:|wrote:|says:|said:" r"|^In article|^Quoted from|^\||^>)"
)


def strip_newsgroup_quoting(text):
    good_lines = [line for line in text.split("\n") if not _QUOTE_RE.search(line)]
    return "\n".join(good_lines)


def strip_newsgroup_footer(text):
    lines = text.strip().split("\n")
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip("-") == "":
            break
    if line_num > 0:
        return "\n".join(lines[:line_num])
    else:
        return text


def fetch_dataset(
        *,
        data_home=None,
        subset="all",
        categories=None,
        shuffle=True,
        random_state=42,
        remove=(),
        download_if_missing=True,
        return_X_y=False,
):
    data_home = get_data_home(data_home=data_home)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    twenty_home = os.path.join(data_home, "20news_home")
    cache = None
    data_idx = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, "zlib_codec")
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * "_")
            print("Cache loading failed")
            print(80 * "_")
            print(e)

    if cache is None:
        if download_if_missing:
            logger.info("Downloading 20news dataset. This may take a few minutes.")
            cache = _download_dataset(
                target_dir=twenty_home, cache_path=cache_path
            )
        else:
            raise OSError("20Newsgroups dataset not found")

    print("Starting to extract the dataset")
    if subset in ("train", "test"):
        data = cache[subset]
    elif subset == "all":
        data_lst = list()
        target = list()
        filenames = list()
        idx = []
        for subset in ("train", "test"):
            data = cache[subset]
            data_lst.extend(data.data)
            target.extend(data.target)
            filenames.extend(data.filenames)
            idx.append(len(data_lst))
        print()
        data_idx = idx
        data.data = data_lst
        data.target = np.array(target)
        data.filenames = np.array(filenames)

    fdescr = load_descr("twenty_newsgroups.rst")

    data.DESCR = fdescr

    if "headers" in remove:
        data.data = [strip_newsgroup_header(text) for text in data.data]
    if "footers" in remove:
        data.data = [strip_newsgroup_footer(text) for text in data.data]
    if "quotes" in remove:
        data.data = [strip_newsgroup_quoting(text) for text in data.data]

    if categories is not None:
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        # Sort the categories to have the ordering of the labels
        labels.sort()
        labels, categories = zip(*labels)
        mask = np.isin(data.target, labels)
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        # searchsorted to have continuous labels
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()
    doc_indices = []
    for filename in data.filenames:
        file_parts = filename.split("/")
        last_part = file_parts[-1]
        doc_indices.append(last_part)

    cate = data.target_names
    labels = []
    for doc in data.target:
        labels.append(cate[doc])

    test_idx = data_idx[0] + 1 if data_idx is not None else 0
    metadata = {"name": "20 Newsgroup", "nr_docs": len(data.data), "nr_labels": len(labels),
                'test_idx': test_idx}
    dataset = Dataset(metadata=metadata, docs=data.data, labels=labels, indices=doc_indices)

    return dataset
