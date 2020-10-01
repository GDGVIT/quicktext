from qtc import *

__all__ = ["Vocab", "_prepare_label"]


class Vocab:
    """
    This class helps in building vocabulary 
    """

    def __init__(self, featurizer, pad_token="@pad@"):
        """
        Constructor function for Vocab class
        Args:
            featurizer (SpacyFeaturizer): Featurizer class, contains untility methods 
                                            to build vocabulary
            pad_token (string): Token used for padding
        Returns:
            None
        """

        self.featurizer = featurizer

    def build(self, texts):
        """
        This function builds vocabulary using the input list of text
        Args:
            texts (List): List of texts
        Return:
            dict: A dictionary of words 
        """

        print("[INFO] Building the vocabulary")

        # Tokenize the text
        self.texts = [self.featurizer.tokenize(text) for text in tqdm(texts)]

        # Build dictionary of word -> index
        self.stoi = {}
        self.stoi["@pad@"] = 0
        index = 1
        for text in self.texts:
            for token in text:
                self.stoi[token] = index
                index += 1

        self.itos = {idx: word for (word, idx) in self.stoi.items()}

    def get_stoi(self):
        """
        This function returns the stoi dictionary
        Args:
            None
        Returns:
            dict: The stoi dictionary
        """

        return self.stoi

    def get_itos(self):
        """
        This function returns the itos dictionary
        Args:
            None
        Returns:
            dict: The itos dictionary
        """

        return self.itos


def _prepare_label(labels):
    """
    This function converts labels to a format that can be 
    used by PyTorch models
    """

    print("[INFO] Preparing the labels ...")

    if str(labels[0].replace(".", "", 1)).isdigit():

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
