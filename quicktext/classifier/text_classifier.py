from quicktext.imports import *
from quicktext.nets.cnn2d import CNN2D, CNN2DFromBase


class TextClassifier:
    """
    This class contains the models and vocab
    """

    def __init__(self, vocab, arch, classes, hparams):
        """
        Constructor class for TextClassifier
        Args:
            vocab (spacy.vocab): Spacy vocabulary class
            arch (string/pl.Lightningmodule): The underlying text classifier model architecture
            hparams (dict): Dictionary of hyper parameters for the underlying model
        Returns:
            None
        """

        self._vocab = vocab
        self._vocab.set_vector("@pad@", vector=np.zeros(self.vocab.vectors.shape[1]))
        self._vocab.set_vector("@oov@", vector=np.zeros(self.vocab.vectors.shape[1]))

        oov_orth = self._vocab["@oov@"].orth
        self.oov_id = self._vocab.vectors.key2row[oov_orth]

        self.classes = classes

        self.tokenizer = Tokenizer(self.vocab)

        if isinstance(arch, pl.LightningModule):
            self._model = arch
        elif isinstance(arch, str):

            INPUT_DIM, EMBEDDING_DIM = self.vocab.vectors.shape
            N_FILTERS = 100
            FILTER_SIZES = [3, 4, 5]
            OUTPUT_DIM = len(self.classes)
            DROPOUT = 0.5
            PAD_IDX = self.vocab.vectors.key2row[self.vocab["@pad@"].orth]
            self._model = CNN2DFromBase(
                INPUT_DIM,
                EMBEDDING_DIM,
                N_FILTERS,
                FILTER_SIZES,
                OUTPUT_DIM,
                DROPOUT,
                PAD_IDX,
            )

    def predict(self, text):
        """
        Classifies text 
        Args:
            text(string): The text to classify
        Returns:
            float: The label of the text
        """

        tokens = self.get_ids(text)
        tokens = torch.tensor(tokens)
        tokens = tokens.unsqueeze(0)
        print(tokens.shape)
        output = self.model(tokens, tokens.shape)
        return output

    @property
    def vocab(self):
        return self._vocab

    @property
    def model(self):
        return self._model

    def get_ids(self, text):
        """
        Returns IDS for tokenized text
        Args:
            text (str): Text to be converted to ids
        Return:
            list: List of ints that map to the rows in embedding layer
        """

        tokens = [token.orth for token in self.tokenizer(text)]
        ids = []
        for token in tokens:
            try:
                id = self._vocab.vectors.key2row[token]
            except KeyError:
                id = self.oov_id

            ids.append(id)

        return ids
