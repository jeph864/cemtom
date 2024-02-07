import os
import zipfile

import gdown
import shutil
import gzip

import requests
from tqdm import tqdm


def download_gdrive(file_id, output=None, quiet=False, cached=False):
    url = f'https://drive.google.com/uc?id={file_id}'
    filename = None
    if cached:
        filename = gdown.cached_download(url, quiet=quiet)
    else:
        filename = gdown.download(url, output, quiet=quiet)
    return filename


def download_file(url, output_dst):
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dst, filename)
    response = requests.get(url, stream=True)
    with open(filepath, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print(f"Download to {filepath}")
    return filepath


def extract(src_file, output_dir='.'):
    filename = src_file.split('/')[-1]
    filepath = os.path.join(output_dir, filename)
    os.remove(filename)

    if filename.endswith('.gz'):
        with gzip.open(src_file, 'rb') as f_in:
            with open(filepath[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif filename.endswith('.zip'):
        with zipfile.ZipFile(src_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        raise IOError('File Format not recognized for extraction')
