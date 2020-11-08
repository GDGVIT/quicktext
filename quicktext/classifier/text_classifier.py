from quicktext.imports import *
from quicktext.nets.cnn2d import CNN2D
from quicktext.nets.bi_lstm import BiLSTM
from quicktext.nets.base import BaseModel


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

        input_dim, embedding_dim = self.vocab.vectors.shape
        output_dim = len(self.classes)
        pad_idx = self.vocab.vectors.key2row[self.vocab["@pad@"].orth]

        hparams["pad_idx"] = pad_idx
        hparams["input_dim"] = input_dim
        hparams["embedding_dim"] = embedding_dim

        if isinstance(arch, BaseModel):
            self._model = arch

        elif isinstance(arch, str):

            if arch == "cnn":

                self._model = CNN2D(output_dim, hparams)

            elif arch == "bilstm":

                self._model = BiLSTM(output_dim, hparams)

            else:
                print("No such architecture exists")

        else:
            print("arch should be string or a torch file duh")

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
        text_length = torch.tensor([tokens.shape[1]])
        print(text_length)
        output = self.model(tokens, text_length)
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
