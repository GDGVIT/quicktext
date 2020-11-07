from quicktext.imports import *


class BaseModel(pl.LightningModule):
    """
    Base model for text classifier architectures
    """

    def __init__(self):
        """
        Constructor function for BaseModel
        Args:
            None
        Returns:
            None
        """
        super().__init__()

    def forward(self, text, seq_len):
        """
        Forward function to define model architecture
        """

        pass

    def training_step(self, batch, batch_idx):
        """
        Train step during model training
        Args:
            batch (dictionary): Dictionary containing texts, seq_lens, labels
            batch_idx (int): Index of batch
        Return:
            dict: 
        """

        text = batch["texts"]
        seq_len = batch["seq_lens"]
        predictions = self(text, seq_len).squeeze(1)

        loss = self.criterion(predictions, batch["labels"].long())

        return {
            "loss": loss,
            "predictions": predictions,
            "label": batch["labels"],
            "log": {"train_loss": loss},
        }

    def validation_step(self, batch, batch_idx):
        """
        Validation step during model training
        Args:
            batch (dictionary): Dictionary containing texts, seq_lens, labels
            batch_idx (int): Index of batch
        Return:
            dict: 
        """
        text = batch["texts"]
        seq_len = batch["seq_lens"]
        predictions = self(text, seq_len).squeeze(1)

        loss = self.criterion(predictions, batch["labels"].long())

        return {
            "val_loss": loss,
            "predictions": predictions,
            "label": batch["labels"],
            "log": {"val_loss": loss},
        }

    def training_epoch_end(self, outputs):
        """
        Training epoch end 
        Args:
            outputs (dictionary): Dictionary containing training statistics collected during epoch
        Return:
            dict: 
        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        y = torch.cat([x["label"] for x in outputs])
        y_hat = torch.cat([x["predictions"] for x in outputs])

        _, preds = torch.max(y_hat, 1)

        acc = accuracy(preds, y)

        print("Training metrics : loss-", avg_loss, ", acc-", acc * 100)

        return {"loss": avg_loss, "train_acc": acc}

    def validation_epoch_end(self, outputs):
        """
        Validation epoch end 
        Args:
            outputs (dictionary): Dictionary containing training statistics collected during epoch
        Return:
            dict: 
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["label"] for x in outputs])
        y_hat = torch.cat([x["predictions"] for x in outputs])

        _, preds = torch.max(y_hat, 1)

        acc = accuracy(preds, y)

        print("Validation metrics : loss-", avg_loss, ", acc-", acc * 100)

        return {"val_loss": avg_loss, "val_acc": acc}

    def configure_optimizers(self):
        """
        Configure the optimizers for training
        """

        return torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad == True]
        )
