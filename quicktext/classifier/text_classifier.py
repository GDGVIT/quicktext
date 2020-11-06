from quicktext.imports import *
from quicktext.nets.cnn2d import CNN2D


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
            self._model = CNN2D(
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

        pass

    @property
    def vocab(self):
        return self._vocab

    @property
    def model(self):
        return self._model
