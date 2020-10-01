from qtc.imports import *

__all__ = ["SpacyFeaturizer"]


class SpacyFeaturizer:
    """
    A class which provides functionality for tokenization 
    and getting feature vectors
    """

    def __init__(self, pad_token="@pad@"):
        """
        Constructor function for SpacyFeaturizer class
        Args:
            pad_token: The token with which text sequence is padded
        Returns:
            None
        """

        self.nlp = en_core_web_md.load(disable=["ner", "tagger"])
        self.tokenizer = Tokenizer(self.nlp.vocab)
        self.pad_token = pad_token

    def tokenize(self, text):
        """
        Tokenizes the text using sPacy tokenizer
        The tokens are in lowercase
        Args:
            text (string):The text to be tokenized
        Returns:
            List: The list of tokens 
        """

        tokens = self.tokenizer(text)
        return [
            token.text.lower()
            for token in token
            if token.text.strip() and not token.is_punct
        ]

    def get_feature_vector(self, _list):
        """
        Returns feature vectors for a list of words
        Args:
            _list (List): List of tokens in sentence
        Returns:
            numpy.ndarray: A feature vector for the seqence
        """

        features = np.array(list(map(self._get_word_vector, _list)))
        return features

    def _get_word_vector(self, token):
        """
        Gets word embeddings for the given word
        Args:
            token (string): The word for which word embedding is required
        Returns:
            numpy.ndarray: The word embedding for the token
        """

        if token == self.pad_token:
            return np.zeros(self.nlp.vocab.vectors.shape[1])
        else:
            return self.nlp.vocab[token].vector
