from quicktext.imports import *
from quicktext.data.classifier_data import TextClassifierData
from quicktext.nets.lightning_module.model_factory import BaseModel


class Trainer:
    """
    This class is used to train the models in quicktext
    """

    def __init__(self, classifier):
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
        self.pl_model = BaseModel(classifier.model, classifier.num_class)

    def load_data(self, data, batch_size):
        """
        This function takes data to return 
        a torch dataloader
        Args:
            data (list): list of tuples
        Returns:
            torch DataLoader:  Torch dataloader
        """

        dataset = TextClassifierData(self.classifier.vocab, data)
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collator)
        return loader

    def fit(self, train_data, val_data, epochs=1, batch_size=32, gpus=0):
        """
        This function trains the model
        Args:
            epochs (int): Number of epoch to train the model
            gpus (int): Number of GPUs to use
        Return:
            None
        """

        train_loader = self.load_data(train_data, batch_size)
        val_loader = self.load_data(val_data, batch_size)

        trainer = pl.Trainer(max_epochs=epochs, gpus=gpus)
        trainer.fit(self.pl_model, train_loader, val_loader)

    def test(self, test_data, batch_size=32, gpus=0):
        """
        This function tests model using test set
        Args:
            gpus (int): The number of gpus
        Returns:
            None
        """

        test_loader = self.load_data(test_data, batch_size)

        trainer = pl.Trainer(gpus=gpus)
        trainer.test(self.classifier.model, test_dataloaders=test_loader)
