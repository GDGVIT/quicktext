from quicktext.imports import *
from quicktext.utils.configuration import read_yaml, merge_dictb_to_dicta

"""
Code for the neural net based on a repo by AnubhavGupta3377
https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
"""


class FastText(nn.Module):
    def __init__(self, n_classes=2, config=None):
        super(FastText, self).__init__()
        self.config = config

        # Embedding Layer
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.embed_size)

        # Hidden Layer
        self.fc1 = nn.Linear(self.config.embed_size, self.config.hidden_size)

        # Output Layer
        self.fc2 = nn.Linear(self.config.hidden_size, self.config.output_size)

    def forward(self, x, text_lengths):

        # x.shape = (seq_len, batch_size)
        x = x.permute(1, 0)
        embedded_sent = self.embeddings(x).permute(1, 0, 2)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)

        return z
