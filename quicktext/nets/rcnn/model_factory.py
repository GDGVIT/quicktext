from quicktext.imports import *
from quicktext.utils.configuration import read_yaml, merge_dictb_to_dicta

"""
Code for the neural net based on a repo by AnubhavGupta3377
https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
"""


class RCNN(nn.Module):
    def __init__(self, num_class=2, config=None):
        super(RCNN, self).__init__()

        main_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(main_dir, "config.yml")
        default_config = read_yaml(config_path)

        config = (
            merge_dictb_to_dicta(default_config, config)
            if config is not None
            else default_config
        )

        # Embedding Layer
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)

        # Bi-directional LSTM for RCNN
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            dropout=config.dropout,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(config.dropout)

        # Linear layer to get "convolution output" to be passed to Pooling Layer
        self.W = nn.Linear(
            config.embedding_dim + 2 * config.hidden_dim, config.hidden_dim_linear,
        )

        # Tanh non-linearity
        self.tanh = nn.Tanh()

        # Fully-Connected Layer
        self.fc = nn.Linear(config.hidden_dim_linear, num_class)

    def forward(self, x, seq_lens):
        # x.shape = (seq_len, batch_size)
        x = x.permute(1, 0)

        embedded_sent = self.embeddings(x)
        # embedded_sent.shape = (seq_len, batch_size, embedding_dim)

        lstm_out, (h_n, c_n) = self.lstm(embedded_sent)
        # lstm_out.shape = (seq_len, batch_size, 2 * hidden_size)

        input_features = torch.cat([lstm_out, embedded_sent], 2).permute(1, 0, 2)
        # final_features.shape = (batch_size, seq_len, embedding_dim + 2*hidden_size)

        linear_output = self.tanh(self.W(input_features))
        # linear_output.shape = (batch_size, seq_len, hidden_size_linear)

        linear_output = linear_output.permute(0, 2, 1)  # Reshaping fot max_pool

        max_out_features = F.max_pool1d(linear_output, linear_output.shape[2]).squeeze(
            2
        )
        # max_out_features.shape = (batch_size, hidden_size_linear)

        max_out_features = self.dropout(max_out_features)
        final_out = self.fc(max_out_features)

        return final_out
