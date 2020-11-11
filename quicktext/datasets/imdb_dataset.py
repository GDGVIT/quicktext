import re
import tarfile
import urllib.request
from quicktext.imports import *

from quicktext.utils.data import load_from_directory


def _download_imdb_dataset(dataset_dir):
    """
    Downloads IMDB dataset to target dir
    Args:
        dataset_dir (str): The folder where data will be stored
    Returns:
        None
    """

    target_dir = "imdb"

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    download_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    tar_dir = os.path.join(dataset_dir, f"{target_dir}.tar.gz")
    urllib.request.urlretrieve(download_url, tar_dir)

    tar_file = tarfile.open(tar_dir)
    tar_file.extractall(dataset_dir)
    tar_file.close()
