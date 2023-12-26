import hashlib
import os
from os import path
from urllib import request


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _pkl_filepath(*args, **kwargs):
    py3_suffix = kwargs.get("py3_suffix", "_py3")
    basename, ext = path.splitext(args[-1])
    basename += py3_suffix
    new_args = args[:-1] + (basename + ext,)
    return path.join(*new_args)


def get_data_home(data_home=None):
    if data_home is None:
        data_home = os.environ.get("CEMTOM_DATA", path.join("~", "cemtom_data"))
    data_home = path.expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


def _fetch(filename=None, url=None, dirname=None):
    file_path = filename if dirname is None else path.join(dirname, filename)
    request.urlretrieve(url, file_path)
    return file_path
