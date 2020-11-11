"""
Functions to clean the 20newsgroups dataset have been taken
from scikit-learn. The design of this dataset utility is also adapted from there

https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/_twenty_newsgroups.py
"""

import re
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


def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    _before, _blankline, after = text.partition("\n\n")
    return after


_QUOTE_RE = re.compile(
    r"(writes in|writes:|wrote:|says:|said:" r"|^In article|^Quoted from|^\||^>)"
)


def strip_newsgroup_quoting(text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    good_lines = [line for line in text.split("\n") if not _QUOTE_RE.search(line)]
    return "\n".join(good_lines)


def strip_newsgroup_footer(text):
    """
    Given text in "news" format, attempt to remove a signature block.
    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    Parameters
    ----------
    text : str
        The text from which to remove the signature block.
    """
    lines = text.strip().split("\n")
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip("-") == "":
            break

    if line_num > 0:
        return "\n".join(lines[:line_num])
    else:
        return text


def get_20newsgroups(
    subset="train",
    shuffle=True,
    random_state=42,
    remove=[],
    return_x_y=False,
    dataset_dir="quicktext_data",
):
    """
    Loads the files from 20 news groups dataset 
    Also the files are processed on demand using arguments from remove
    Args:
        subset (str): The subset of data to return
        shuffle (boolean): Shuffles data if set to True
        random_state (int): Random state kept while splitting data into train val and test
        remove (list): List of metadata to remove from the 20newsgroups dataset
        return_x_y (bool): If True returns data in form (text, label)
    Returns:
        dict: With keys data, target, target_names
        or it could return a list of tuples of form (text, label)
    """

    target_dir = os.path.join(dataset_dir, "20_newsgroups")

    _download_20newsgroups(dataset_dir)

    data = load_from_directory(target_dir)

    data.data = [text.decode("latin-1") for text in data.data]

    if "headers" in remove:
        data.data = [strip_newsgroup_header(text) for text in data.data]
    if "footers" in remove:
        data.data = [strip_newsgroup_footer(text) for text in data.data]
    if "quotes" in remove:
        data.data = [strip_newsgroup_quoting(text) for text in data.data]

    if return_x_y:
        data = [(data.data[idx], data.target[idx]) for idx in range(len(data.target))]

    return data
