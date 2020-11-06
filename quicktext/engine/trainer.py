from quicktext.imports import *
from quicktext.data.classifier_data import TextClassifierData


class Trainer:
    """
    This class is used to train the models in quicktext
    """

    def __init__(self, classifier, train, val, test, batch_size=32):
        """
        Constructor function for Trainer class
        Args:
            classifier (TextClassifier): Text classifier class
            train (list): List of tuples of form (text, label) for training
            val (list): List of tuples of form  (text, label) for validation
            test (list): List of tuples of form  (text, label) for test
        Returns:   
            None
        """

        self.classifier = classifier
        self.batch_size = batch_size

        self.train_loader = self.load_data(train)
        self.val_loader = self.load_data(val)
        self.test_loader = self.load_data(test)

    def load_data(self, data):
        """
        This function takes docs and labels to return 
        a torch dataloader
        Args:
            data (list): list of tuples
        Returns:
            torch DataLoader:  Torch dataloader
        """

        dataset = TextClassifierData(self.classifier.vocab, data)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=data.get_batch
        )
        return loader

    def fit(self, epochs, gpus):
        trainer = pl.Trainer(max_epochs=epochs, gpus=gpus)
        trainer.fit(self.classifier.model, self.train_loader ,self.val_loader)