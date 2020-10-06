from quicktext.imports import *


class Vocab:
    """
    This class helps in building vocabulary 
    """

    def __init__(self, min_freq=3, pad_token="<pad>"):
        """
        Constructor function for Vocab class
        Args:
            pad_token (string): Token used for padding
        Returns:
            None
        """

        self._pad_token = pad_token
        self._min_freq = min_freq

    def build(self, texts):
        """
        This function builds vocabulary using the input list of text
        Args:
            texts (List): List of tokenized texts
        Return:
            None 
        """

        print("[INFO] Building the vocabulary")

        # Counter object
        counter = Counter()

        for text in texts:
            for token in text:
                counter[token] += 1

        # Build dictionary of word -> index
        words = [key for key in counter if counter[key] >= self._min_freq]

        self._stoi = {word: idx + 1 for idx, word in enumerate(words)}
        self._stoi[self._pad_token] = 0

        self._itos = {idx: word for (word, idx) in self._stoi.items()}

    @property
    def stoi(self):
        return self._stoi

    @property
    def itos(self):
        return self._itos

    @property
    def size(self):
        return len(self._stoi)
