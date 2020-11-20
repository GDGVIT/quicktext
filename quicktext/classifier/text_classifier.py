import en_core_web_md

from quicktext.imports import *
from quicktext.nets.cnn2d.model_factory import CNN2D
from quicktext.nets.lstm.model_factory import BiLSTM
from quicktext.nets.fasttext.model_factory import FastText
from quicktext.nets.rcnn.model_factory import RCNN
from quicktext.nets.seq2seq.model_factory import Seq2SeqAttention
from quicktext.nets.lightning_module.model_factory import BaseModel


class TextClassifier:
    """
    This class contains the models and vocab
    """

    def __init__(self, n_classes, arch="cnn2d", vocab=None, config={}):
        """
        Constructor class for TextClassifier
        Args:
            vocab (spacy.vocab): Spacy vocabulary class
            arch (string/pl.Lightningmodule): The underlying text classifier model architecture
            config (dict): Dictionary of hyper parameters for the underlying model
        Returns:
            None
        """

        if isinstance(vocab, Vocab):
            self._vocab = vocab
        else:
            self._vocab = en_core_web_md.load().vocab

        self._vocab.set_vector("@pad@", vector=np.zeros(self.vocab.vectors.shape[1]))
        self._vocab.set_vector("@oov@", vector=np.zeros(self.vocab.vectors.shape[1]))

        oov_orth = self._vocab["@oov@"].orth
        self.oov_id = self._vocab.vectors.key2row[oov_orth]

        self.tokenizer = Tokenizer(self.vocab)

        input_dim, embedding_dim = self.vocab.vectors.shape
        output_dim = n_classes
        pad_idx = self.vocab.vectors.key2row[self.vocab["@pad@"].orth]

        config["pad_idx"] = pad_idx
        config["input_dim"] = input_dim
        config["embedding_dim"] = embedding_dim

        if isinstance(arch, BaseModel):
            self._model = arch

        elif isinstance(arch, str):

            if arch == "cnn2d":

                self._model = CNN2D(output_dim, config)

            elif arch == "fasttext":
                self._model = FastText(output_dim, config)

            elif arch == "seq2seq":
                self._model = Seq2SeqAttention(output_dim, config)

            elif arch == "rcnn":
                self._model = RCNN(output_dim, config)

            elif arch == "bilstm":

                self._model = BiLSTM(output_dim, config)

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

    def save(self, directory):
        """
        Saves PyTorch model and sPacy vocab in directory
        if directory doesnt exist its created
        else contents overwritten
        Args:
            directory (str):  Directory where models will be stored
        Returns:
            None
        """

        if not os.path.exists(directory):
            os.mkdir(directory)

        self._vocab.to_disk(os.path.join(directory, "spacy_vocab"))

        torch.save(self._model, os.path.join(directory, "torch_model"))

    def from_pretrained(self, directory):
        """
        Loads PyTorch model and sPacy vocab from directory
        Args:
            directory (str): Directory where models are stored
        Returns:
            None
        """

        if not os.path.exists(directory):
            print("The path does not exists")

        self._vocab = Vocab().from_disk(os.path.join(directory, "spacy_vocab"))
        self._model = torch.load(os.path.join(directory, "torch_model"))
