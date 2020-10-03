from quicktext.imports import *
from quicktext.text.doc import Doc
from quictext.text.vocab import Vocab
from quicktext.data.classifier_data import TextClassifierData


class SpacyTextProcessor:
    """
    This class is used for processing the text data
    It does the following processes
        - Tokenizes the data
        - Build vocabulary
        - Creates doc objects
    """

    def __init___(self, model="en_core_web_md"):
        """
        This is the constructor for SpacyTextProcessor class
        This function loads the tokenizers
        Args:
            model(string): The spacy model to use
            min_word_freq (int): The minimum frequency of words in training data
                                so that they can be added to vocabulary
        Retuns:
            None
        """

        self.nlp = spacy.load(model, disable=["ner", "tagger"])
        self.tokenizer = Tokenizer(self.nlp.vocab)

        self.vocab = Vocab()

    def get_dataset(self, texts, labels, test_size=0.2, val_size=0.2, min_freq=10):
        """
        This function processes the dataset by
            - Splitting dataset into train, val and test
            - Tokenizing the text 
            - Building vocabulary using train
        Args:
            texts (list): List of texts
            labels (list): List of labels
        Returns:
            dict: Dictionary containing processed data
        """

        # Split dataset

        train_texts, test_texts, train_labels, text_labels = train_test_split(
            texts, labels, test_size
        )

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=(1 - test_size) * val_size
        )

        tokenized_texts = self.tokenize(train_texts)

        # Build vocabulary
        self.vocab.build(tokenized_texts, min_freq)

        # Prepare dataset

    def build_dataset(self, texts, labels):
        """
        This function builds a torch dataset 
        which returns list of Doc objects
        Args:
            texts (list): List of texts
            labels (list): List of labels 
        Return:
            torch.utils.data.Dataset: A torch dataset object
        """

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

        tokens = [
            token.text.lower()
            for token in tokens
            if token.text.strip() and not token.is_punct
        ]

        doc = Doc(text)
        doc.tokens = tokens

        return doc
