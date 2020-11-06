from quicktext.imports import *
from quicktext.data.utils import pad_tokens
from quicktext.featurizers import SpacyFeaturizer

__all__ = ["TextClassifierData"]


class TextClassifierData2(Dataset):
    """
    This class provides labels and tokenized and vectorized text
    """

    def __init__(self, docs, labels, featurizer=None):
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
        self.featurizer = featurizer

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

        # Create tensors for labels and seq_lens
        labels = torch.tensor(labels)
        seq_lens = torch.tensor(seq_lens)

        if self.featurizer:
            texts = [doc.tokens for doc in docs]
            texts = [self.featurizer.get_feature_vector(tokens) for tokens in texts]
            texts = np.stack(texts)
            texts = torch.from_numpy(texts)

            batch = {
                "inputs": texts,
                "labels": labels,
                "seq_lens": seq_lens,
            }

        else:
            ids = [np.array(doc.ids) for doc in docs]
            ids = np.stack(ids)
            ids = torch.from_numpy(ids)

            batch = {"inputs": ids, "labels": labels, "seq_lens": seq_lens}

    return batch


class TextClassifierData(Dataset):
    """
    This class provides labels and tokenized and vectorized text
    """

    def __init__(self, vocab, data):
        """
        Constructor function for TextClassifierData class
        Args:
            data (List): List of tuples of form (text, label)
            vocab (spacy.vocab): spaCy vocabulary class
        Returns:
            None
        """
        self.vocab = vocab
        self.tokenizer = Tokenizer(vocab)
        self.data = data

        oov_orth = self.vocab["@oov@"].orth
        self.oov_id = self.vocab.vectors.key2row[oov_orth]

        pad_orth = self.vocab["@pad@"].orth
        self.pad_id = self.vocab.vectors.key2row[pad_orth]

    def __len__(self):
        """
        Returns the total length of the dataset
        Args:
            None
        Return:
            int: Total length of the dataset
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns an item from dataset
        Args:
            idx (int): The index of the data item in dataset
        Returns:
            dict: A dictionary with ids and label
        """

        text, label = self.data[idx]
        ids = self.get_ids(text)

        return {"ids": ids, "label": label}

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
        ids = [item["ids"] for item in batch]
        labels = [item["label"] for item in batch]

        # Sort the list
        ids, labels = map(
            list,
            zip(
                *sorted(
                    zip(ids, labels), key=lambda _tuple: len(_tuple[0]), reverse=True,
                )
            ),
        )

        max_len = len(ids[0])

        # Initialize seq len list
        seq_lens = []
        new_ids = []
        for id in ids:

            _len = len(id)
            pad_len = max_len - _len

            if pad_len < 0:
                id = id[:max_len]
            else:
                id = np.pad(
                    id, (0, pad_len), "constant", constant_values=self.pad_id
                ).tolist()

            new_ids.append(id)

            seq_lens.append(_len if _len < max_len else max_len)

        labels = torch.tensor(labels)
        seq_lens = torch.tensor(seq_lens)
        text = np.stack(new_ids)
        text = torch.from_numpy(text)

        return {"labels": labels, "seq_lens": seq_lens, "texts": text}

    def get_ids(self, text):
        """
        Maps tokens to ids in embedding layer
        Args:
            text (string): Text to be converted to ids
        Returns:
            list: A list of corresponding token ids
        """

        tokens = [token.orth for token in self.tokenizer(text)]
        ids = []
        for token in tokens:
            try:
                id = self.vocab.vectors.key2row[token]
            except KeyError:
                id = self.oov_id

            ids.append(id)

        return ids
