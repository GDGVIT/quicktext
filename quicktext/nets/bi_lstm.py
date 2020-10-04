from quicktext.imports import *
from quicktext.featurizers.spacy_featurizer import SpacyFeaturizer

"""
Code for the neural net based on a repo by bentrevett
https://github.com/bentrevett/pytorch-sentiment-analysis
"""

__all__ = ["BiLSTM"]


class BiLSTM(pl.LightningModule):
    def __init__(
        self,
        output_dim,
        pretrained_vectors=True,
        vocab_size=None,
        embedding_dim=300,
        hidden_dim=128,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
        pad_idx=0,
    ):
        super().__init__()

        self.pretrained_vectors = pretrained_vectors

        if not self.pretrained_vectors:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=pad_idx
            )

        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.CrossEntropyLoss()

        self.featurizer = SpacyFeaturizer("en_core_web_md")

    def forward(self, vectors, seq_lengths):

        if not self.pretrained_vectors:
            # text = [sent len, batch size]

            vectors = self.dropout(self.embedding(vectors))

        # embedded = [ batch size, sent len, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            vectors, seq_lengths, batch_first=True
        )

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)

    def training_step(self, batch, batch_idx):

        # Convert seq_lens and labels to tensors

        text_lengths = torch.tensor(batch["seq_lens"])
        labels = torch.tensor(batch["labels"])

        # Need to convert docs to tensor of tokens or ids
        docs = batch["docs"]

        if self.pretrained_vectors:
            text = [doc.tokens for doc in docs]
            text = [self.featurizer.get_feature_vector(tokens) for tokens in text]
            text = np.stack(text)
            text = torch.from_numpy(text)

        else:
            text = [np.array(doc.ids) for doc in docs]
            text = np.stack(text)
            text = torch.from_numpy(text)

        text = text.float()

        predictions = self(text, text_lengths).squeeze(1)

        loss = self.criterion(predictions, labels.long())

        return {
            "loss": loss,
            "predictions": predictions,
            "label": labels,
            "log": {"train_loss": loss},
        }

    def validation_step(self, batch, batch_idx):

        # Convert seq_lens and labels to tensors

        text_lengths = torch.tensor(batch["seq_lens"])
        labels = torch.tensor(batch["labels"])

        # Need to convert docs to tensor of tokens or ids
        docs = batch["docs"]

        if self.pretrained_vectors:
            text = [doc.tokens for doc in docs]
            text = [self.featurizer.get_feature_vector(tokens) for tokens in text]
            text = np.stack(text)
            text = torch.from_numpy(text)

        else:
            text = [np.array(doc.ids) for doc in docs]
            text = np.stack(text)
            text = torch.from_numpy(text)

        text = text.float()

        predictions = self(text, text_lengths).squeeze(1)

        loss = self.criterion(predictions, labels.long())

        return {
            "val_loss": loss,
            "predictions": predictions,
            "label": labels,
            "log": {"val_loss": loss},
        }

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        y = torch.cat([x["label"] for x in outputs])
        y_hat = torch.cat([x["predictions"] for x in outputs])

        _, preds = torch.max(y_hat, 1)

        acc = accuracy(preds, y)

        print("Training metrics : loss-", avg_loss, ", acc-", acc * 100)

        return {"loss": avg_loss, "train_acc": acc}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["label"] for x in outputs])
        y_hat = torch.cat([x["predictions"] for x in outputs])

        _, preds = torch.max(y_hat, 1)

        acc = accuracy(preds, y)

        print("Validation metrics : loss-", avg_loss, ", acc-", acc * 100)

        return {"val_loss": avg_loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
