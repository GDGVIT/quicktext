from quicktext.imports import *
from quicktext.nets.base import BaseModel

"""
Code for the neural net based on a repo by bentrevett
https://github.com/bentrevett/pytorch-sentiment-analysis
"""

__all__ = ["CNN2D"]


class CNN2DFromBase(BaseModel):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):

        super(CNN2DFromBase, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text, seq_len):

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


class CNN2D(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text):

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def training_step(self, batch, batch_idx):

        text = batch["texts"]

        predictions = self(text).squeeze(1)

        loss = self.criterion(predictions, batch["labels"].long())

        return {
            "loss": loss,
            "predictions": predictions,
            "label": batch["labels"],
            "log": {"train_loss": loss},
        }

    def validation_step(self, batch, batch_idx):

        text = batch["texts"]

        predictions = self(text).squeeze(1)

        loss = self.criterion(predictions, batch["labels"].long())

        return {
            "val_loss": loss,
            "predictions": predictions,
            "label": batch["labels"],
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
        return torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad == True]
        )
