from quicktext.imports import *
from quicktext.text.vocab import Vocab
from quicktext.text.doc import Doc


class Corpus:
    """
    This class builds the corpus for the dataset
    This includes building the vocabulary
    Tokenizing the text and creating doc objects
    """

    def __init__(self, train_data, test_data, val_data, language="en", min_freq=3):
        """
        Constructor class for Corpus class
        Args:
            train_data (str): Path to train csv file
            test_data (str): Path to test csv file
            val _data (str): Path to val csv file
            language (str): The language of the dataset
        Returns:
            None
        """

        nlp = en_core_web_md.load(disable=["ner", "tagger"])
        self.tokenizer = Tokenizer(nlp.vocab)

        # Initialize vocab
        self.vocab = Vocab(min_freq)

        # Build vocabulary
        self.build_vocabulary(train_data)

        # Build bundles
        self.train_data = self.build_bundle(train_data)
        self.test_data = self.build_bundle(test_data)
        self.val_data = self.build_bundle(val_data)

    def build_vocabulary(self, train_path):
        """
        This function builds vocabulary using the training data
        Args:
            train_path (str): Path to training data csv file
        Returns:
            None
        """

        df = pd.read_csv(train_path)
        train_texts = df["text"].tolist()

        tokenized_texts = [self.tokenize(text) for text in train_texts]

        # Build vocabulary
        self.vocab.build(tokenized_texts)

    def build_bundle(self, path):
        """
        This function takes a csv dataset of text and labels
        and returns a tuple of (doc lists, labels)
        
        A doc is an object of class Doc
        This class is used to hold text and its corresponding tokens and ids

        Args:
            path (str): Path to csv file of dataset
        Returns:
            list: List of doc objects
            list: List of labels
        """

        df = pd.read_csv(path)
        texts = df["text"].tolist()
        labels = df["label"].tolist()

        docs = [self.build_doc(text) for text in texts]

        return docs, labels

    def build_doc(self, text):
        """
        This function builds a doc object
        The doc object contains text, tokens, ids
        Args:
            text (str): The text for which doc object will be built
        Returns:
            Doc: Object of class Doc
        """

        doc = Doc(text)
        doc.tokens = self.tokenize(text)
        doc.ids = [
            self.vocab.stoi[token] if token in self.vocab.stoi else self.pad_token
            for token in doc.tokens
        ]
        return doc

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

        return tokens
