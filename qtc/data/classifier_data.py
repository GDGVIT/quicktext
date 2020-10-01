from qtc.imports import *
from qtc.data.utils import Vocab, _prepare_labels
from qtc.featurizers import SpacyFeaturizer

__all__ = ["TextClassifierData"]


class TextClassifierData(Dataset):
    """
    This class provides labels and tokenized and vectorized text
    """

    def __init__(self, texts, labels, model="en_core_web_md"):
        """
        Constructor function for TextClassifierData class
        Args:
            texts (List): List of texts in dataset
            labels (List): List of labels in dataset
            model (string): List of sPacy model to use
        Returns:
            None
        """

        self._texts = texts
        self._labels = labels

        self.featurizer = SpacyFeaturizer()

        self.vocab = Vocab(self.featurizer)

        # Prepare labels
        self._labels, _, _ = _prepare_labels(self._labels)

        # Build vocabulary
        self.vocab.build(self._texts)

        self._idx_texts = self.vocab.get_tokenized_texts()

        self.stoi = self.vocab.get_stoi()
        self.itos = self.vocab.get_itos()

    def __len__(self):
        """
        Returns the total length of the dataset
        Args:
            None
        Return:
            int: Total length of the dataset
        """

        return len(self._texts)

    def __getitem__(self, idx):
        """
        Returns an item from dataset
        Args:
            idx (int): The index of the data item in dataset
        Returns:
            dict: A dictionary with text and label
        """

        # Get text and label
        _idx_text = self._idx_texts[idx]
        _label = self._labels[idx]

        return {"idx_text": _idx_text, "label": _label}

    def get_batch(self, batch):
        """
        Collate function for PyTorch dataloaders
        This function is required to pad the sequences in batch
        Args:
            batch (dict): The dictionary of texts and labels
        Returns:
            dict: This dictionary contains vectorized text input, labels 
                    and sequence lengths which are required in RNN
        """

        texts = []
        seq_lens = []
        labels = []

        for item in batch:
            _idx_text = item["idx_text"]
            _label = item["label"]
            _len = len(_idx_text)

            pad_len = self.max_len - len(_idx_text)

            if pad_len < 0:
                _idx_text = _idx_text[: self.max_len]
            else:
                _idx_text = np.pad(
                    _idx_text, (0, pad_len), "constant", constant_values=0
                ).tolist()

            _text = [self.itos[_idx] for _idx in _idx_text]

            _feature = self.featurizer.get_feature_vector(_text)

            texts.append(_feature)
            labels.append(_label)

            seq_lens.append(_len if _len < self.max_len else self.max_len)

        texts = np.stack(texts)
        texts = torch.from_numpy(texts)

        return {
            "texts": texts,
            "labels": torch.tensor(labels),
            "seq_lens": torch.tensor(seq_lens),
        }
