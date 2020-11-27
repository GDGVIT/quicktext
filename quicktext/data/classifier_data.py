from quicktext.imports import *


__all__ = ["TextClassifierData"]


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

    def collator(self, batch):
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
        label = [item["label"] for item in batch]

        # Sort the list
        ids, label = map(
            list,
            zip(
                *sorted(
                    zip(ids, label), key=lambda _tuple: len(_tuple[0]), reverse=True,
                )
            ),
        )

        max_len = len(ids[0])

        # Initialize seq len list
        text_lengths = []
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

            text_lengths.append(_len if _len < max_len else max_len)

        label = torch.tensor(label)
        text_lengths = torch.tensor(text_lengths)
        text = np.stack(new_ids)
        text = torch.from_numpy(text)

        return {"label": label, "text_lengths": text_lengths, "text": text}

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
