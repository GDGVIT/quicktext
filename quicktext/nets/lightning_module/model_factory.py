from quicktext.imports import *


class BaseModel(pl.LightningModule):
    """
    Base model for text classifier architectures
    """

    def __init__(self, model, num_class):
        """
        Constructor function for BaseModel
        Args:
            None
        Returns:
            None
        """
        super().__init__()
        self.epoch_count = -1
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = Accuracy()
        self.precision_score = Precision(num_class)
        self.recall_score = Recall(num_class)

    def forward(self, text, text_lengths):
        """
        Forward function to define model architecture
        """

        return self.model(text, text_lengths)

    def training_step(self, batch, batch_idx):
        """
        Train step during model training
        Args:
            batch (dictionary): Dictionary containing texts, seq_lens, labels
            batch_idx (int): Index of batch
        Return:
            dict: 
        """

        prediction, loss = self._shared_eval(batch, batch_idx)

        self.log("train_loss", loss)

        return {
            "loss": loss,
            "prediction": prediction,
            "label": batch["label"],
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

        prediction, loss = self._shared_eval(batch, batch_idx)

        self.log("val_loss", loss)

        return {
            "val_loss": loss,
            "prediction": prediction,
            "label": batch["label"],
        }

    def test_step(self, batch, batch_idx):
        """
        Test step during model testing
        Args:
            batch (dictionary): Dictionary containing texts, seq_lens, labels
            batch_idx (int): Index of batch
        Return:
            dict: 
        """

        prediction, loss = self._shared_eval(batch, batch_idx)

        self.log("test_loss", loss)

        return {
            "test_loss": loss,
            "prediction": prediction,
            "label": batch["label"],
        }

    def _shared_eval(self, batch, batch_idx):

        text = batch["text"]
        text_lengths = batch["text_lengths"].cpu()

        prediction = self(text, text_lengths).squeeze(1)
        label = batch["label"].long()

        loss = self.criterion(prediction, label)

        return prediction, loss

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
        y_hat = torch.cat([x["prediction"] for x in outputs])

        _, preds = torch.max(y_hat, 1)

        acc = self.accuracy(preds, y)
        pre = self.precision_score(preds, y)
        rec = self.recall_score(preds, y)

        print("\nTraining metrics:")
        print(
            "Loss: {:.3f} Accuracy: {:.3f} ".format(
                avg_loss.item(), acc.item() * 100
            )
        )

        print("Precision: {:.3f} Recall: {:.3f}".format(pre.item(), rec.item()))

        return None

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
        y_hat = torch.cat([x["prediction"] for x in outputs])

        _, preds = torch.max(y_hat, 1)

        acc = self.accuracy(preds, y)
        pre = self.precision_score(preds, y)
        rec = self.recall_score(preds, y)

        if self.epoch_count > -1:
            print("-" * 50)
            print("Epoch {} statistics".format(self.epoch_count))
            print("\nValidation metrics : ")
            print(
                "Loss: {:.3f} Accuracy: {:.3f} ".format(
                    avg_loss.item(), acc.item() * 100
                )
            )

            print("Precision: {:.3f} Recall: {:.3f}".format(pre.item(), rec.item()))

        else:
            print("Validation sanity fit complete")

        self.epoch_count += 1

        return None

    def configure_optimizers(self):
        """
        Configure the optimizers for training
        """

        return torch.optim.Adam(
            [param for param in self.parameters() if param.requires_grad == True]
        )
