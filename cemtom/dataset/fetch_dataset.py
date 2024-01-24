from ._20newgroup import fetch_dataset as fetch20


def fetch_dataset(name="20Newsgroup", remove=None):
    if name == "20Newsgroup":
        return fetch20(remove=remove)
    else:
        raise ValueError("Dataset does not exist")
