import tarfile
import urllib.request
from quicktext.imports import *

from quicktext.utils.data import load_from_directory


def _download_20newsgroups(dataset_dir):
    """
    Downloads the 20newsgroups data to target dir
    Args:
        target_dir (str): Folder where dataset will be stored
    Returns:
        None 
    """

    target_dir = "20_newsgroups"

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    download_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz"

    tar_dir = os.path.join(dataset_dir, f"{target_dir}.tar.gz")
    urllib.request.urlretrieve(download_url, tar_dir)

    tar_file = tarfile.open(tar_dir)
    tar_file.extractall(dataset_dir)
    tar_file.close()
