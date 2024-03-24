from ._20newgroup import fetch_dataset as fetch20

__datasets__ = [
    "20ng",
    "nips",
    "kos",
    "rcv1",
    "bbc_news",
    "m10",
    "dblp"
]


def fetch_dataset(name="20Newsgroup", remove=None):
    if name == "20Newsgroup":
        return fetch20(remove=remove)
    else:
        raise ValueError("Dataset does not exist")
