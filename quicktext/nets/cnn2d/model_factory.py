from quicktext.imports import *
from quicktext.nets.base import BaseModel
from quicktext.utils.configuration import read_yaml, merge_dictb_to_dicta

"""
Code for the neural net based on a repo by bentrevett
https://github.com/bentrevett/pytorch-sentiment-analysis
"""

class CNN2D(nn.Module):
    def __init__(self, output_dim, hparams):

        super().__init__()

        main_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(main_dir, "config.yml")
        default_hparams = read_yaml(config_path)

        hparams = merge_dictb_to_dicta(default_hparams, hparams)

        self.embedding = nn.Embedding(
            hparams.vocab_size, hparams.embedding_dim, padding_idx=hparams.pad_idx
        )

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=hparams.n_filters,
                    kernel_size=(fs, hparams.embedding_dim),
                )
                for fs in hparams.filter_sizes
            ]
        )

        self.fc = nn.Linear(len(hparams.filter_sizes) * hparams.n_filters, output_dim)

        self.dropout = nn.Dropout(hparams.dropout)
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
