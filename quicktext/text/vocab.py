from quicktext.imports import *


class Vocab:
    """
    This class helps in building vocabulary 
    """

    def __init__(self, pad_token="<pad>"):
        """
        Constructor function for Vocab class
        Args:
            pad_token (string): Token used for padding
        Returns:
            None
        """

        self._pad_token = pad_token

    def build(self, texts, min_freq):
        """
        This function builds vocabulary using the input list of text
        Args:
            texts (List): List of tokenized texts
        Return:
            None 
        """

        print("[INFO] Building the vocabulary")

        # Build dictionary of word -> index
        self._stoi = {}
        self._stoi[self._pad_token] = 0
        index = 1
        for text in texts:
            for token in text:
                self._stoi[token] = index
                index += 1

        self._itos = {idx: word for (word, idx) in self._stoi.items()}

    @property
    def stoi(self):
        return self._stoi

    @property
    def itos(self):
        return self._itos
