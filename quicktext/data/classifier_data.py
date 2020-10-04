from quicktext.imports import *
from quicktext.data.utils import Vocab, _prepare_labels, pad_tokens
from quicktext.featurizers import SpacyFeaturizer

__all__ = ["TextClassifierData"]


class TextClassifierData(Dataset):
    """
    This class provides labels and tokenized and vectorized text
    """

    def __init__(self, docs, labels):
        """
        Constructor function for TextClassifierData class
        Args:
            texts (List): List of texts in dataset
            labels (List): List of labels in dataset
        Returns:
            None
        """

        self.labels = labels
        self.docs = docs

    def __len__(self):
        """
        Returns the total length of the dataset
        Args:
            None
        Return:
            int: Total length of the dataset
        """

        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns an item from dataset
        Args:
            idx (int): The index of the data item in dataset
        Returns:
            dict: A dictionary with text and label
        """

        # Get doc and label
        doc = self.docs[idx]
        label = self.labels[idx]

        return {"doc": doc, "label": label}

    def get_batch(self, batch):
        """
        Collate function for PyTorch dataloaders
        This function is required to pad the sequences in batch
        Args:
            batch (dict): The dictionary of docs and labels
        Returns:
            dict: This dictionary contains vectorized text input, labels 
                    and sequence lengths which are required in RNN
        """

        # Retrieve data from batch
        docs = [item["doc"] for item in batch]
        labels = [item["label"] for item in batch]

        # Sort the list
        docs, labels = map(
            list,
            zip(
                *sorted(
                    zip(docs, labels),
                    key=lambda _tuple: len(_tuple[0].tokens),
                    reverse=True,
                )
            ),
        )

        max_len = len(docs[0].tokens)

        # Initialize seq len list
        seq_lens = []
        for doc in docs:

            _len = len(doc.tokens)
            pad_len = max_len - _len

            if pad_len < 0:
                doc.ids = doc.ids[:max_len]

                doc.tokens = doc.tokens[:max_len]
            else:
                doc.ids = np.pad(
                    doc.ids, (0, pad_len), "constant", constant_values=0
                ).tolist()

                doc.tokens = pad_tokens(doc.tokens, max_len)

            seq_lens.append(_len if _len < max_len else max_len)

        return {"docs": docs, "labels": labels, "seq_lens": seq_lens}
