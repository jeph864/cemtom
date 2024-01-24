from .dataset import Dataset
from datasets import load_dataset


def fetch():
    wikitext = load_dataset('wikitext', 'wikitext-103-raw-v1')

