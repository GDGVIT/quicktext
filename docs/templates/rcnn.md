# RCNN

Recurrent Convolution Neural Network combine both recurrent neural network and convolutional neural network to better capture the semantics of the text

# Config

| Parameters                                        | Explanation                           |
|---------------------------------------------------|---------------------------------------|
| vocab_size (int, optional, defaults to 26000)     | Vocabulary size the BiLSTM model      |
| embedding_dim (int, optional, defaults to 300)    | Dimensionality of embedding layer     |
| hidden_dim (int, optional, defaults to 64)        | Dimensionality of hidden layer        |
| n_layers (int, optional, defaults to 1)           | Number of hidden layers in LSTM       |
| hidden_dim_linear (int, optional, defaults to 64) | Dimensionality of hidden linear layer |
| dropout (float, optional, defaults to 0.8)        | Dropout probability for dropout layer |