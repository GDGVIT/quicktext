from quicktext.imports import *


class DataIterator(DataLoader):
    """
    This class builds dataloaders
    Collate function needs to be used for dataloaders
    """

    def __init__(self, data, batch_size):
        """
        Constructor function for DataIterator class
        """
        super(DataIterator, self).__init__(
            data, batch_size=batch_size, collate_fn=data.get_batch
        )
