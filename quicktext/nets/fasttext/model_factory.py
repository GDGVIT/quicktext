from quicktext.imports import *
from quicktext.utils.configuration import read_yaml, merge_dictb_to_dicta

"""
Code for the neural net based on a repo by AnubhavGupta3377
https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
"""


class FastText(nn.Module):
    def __init__(self, num_class=2, config=None):
        super(FastText, self).__init__()

        main_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(main_dir, "config.yml")
        default_config = read_yaml(config_path)

        config = (
            merge_dictb_to_dicta(default_config, config)
            if config is not None
            else default_config
        )

        # Embedding Layer
        self.embeddings = nn.Embedding(config.vocab_size, config.embed_size)

        # Hidden Layer
        self.fc1 = nn.Linear(config.embed_size, config.hidden_size)

        # Output Layer
        self.fc2 = nn.Linear(config.hidden_size, num_class)

    def forward(self, x, text_lengths):

        # x.shape = (seq_len, batch_size)
        x = x.permute(1, 0)
        embedded_sent = self.embeddings(x).permute(1, 0, 2)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)

        return z
