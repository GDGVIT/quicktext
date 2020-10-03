from quicktext.imports import *


class Doc:
    """
    This class provides a container to store the tokens and 
    ids for text in the dataset
    """

    def __init__(self, text):
        """
        Contructor function for Doc class
        Args:
            text (string): The text for the Doc class
        Returns:
            None
        """

        self._text = text

        self._tokens = None
        self._ids = None

    @property
    def ids(self):
        return self._ids

    @ids.setter
    def ids(self, new_ids):
        if isinstance(new_ids, list):
            self._ids = new_ids

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, new_tokens):
        if isinstance(new_tokens, list):
            self._tokens = new_tokens
