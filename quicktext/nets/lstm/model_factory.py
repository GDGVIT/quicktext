from quicktext.imports import *
from quicktext.utils.configuration import read_yaml, merge_dictb_to_dicta

"""
Code for the neural net based on a repo by bentrevett
https://github.com/bentrevett/pytorch-sentiment-analysis
"""


class BiLSTM(nn.Module):
    def __init__(self, output_dim, config=None):
        super().__init__()

        main_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(main_dir, "config.yml")
        default_config = read_yaml(config_path)

        config = (
            merge_dictb_to_dicta(default_config, config)
            if config is not None
            else default_config
        )

        self.embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim, padding_idx=config.pad_idx
        )

        self.rnn = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.n_layers,
            bidirectional=config.bidirectional,
            dropout=config.dropout,
        )

        self.fc = nn.Linear(config.hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(config.dropout)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text, text_lengths):

        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True
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
