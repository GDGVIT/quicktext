from quicktext.imports import *

__all__ = ["prepare_labels"]


def prepare_labels(labels):
    """
    This function converts labels to a format that can be 
    used by PyTorch models
    """

    print("[INFO] Preparing the labels ...")

    if str(labels[0]).replace(".", "", 1).isdigit():

        # Convert to int
        labels = list(map(int, labels))
        labels = list(map(str, labels))

    unique_labels = list(set(labels))

    label2idx = {}
    idx2label = {}

    for idx, label in enumerate(unique_labels):
        label2idx[label] = float(idx)
        idx2label[float(idx)] = label

    labels = [label2idx[_label] for _label in labels]

    return labels, idx2label, label2idx


def pad_tokens(tokens, max_len, pad_token="<pad>"):
    """
    This function pads the tokens list
    with the pad_token
    Args:
        tokens (list): List of tokens
        max_len (int): Maximum length of sequence
        pad_token (str): The padding token
    Returns:
        list (list): List of padded sequence of length max_len
    """

    pad_len = max_len - len(tokens)
    padded_list = tokens + [pad_token] * pad_len
    return padded_list
