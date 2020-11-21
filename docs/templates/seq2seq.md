# Seq2Seq With Attention

## Overview

A sequence to sequence model with attention mechanism for text classification

## Config

| Parameters                                     | Explanation                           |
|------------------------------------------------|---------------------------------------|
| vocab_size (int, optional, defaults to 26000)  | Vocabulary size the BiLSTM model      |
| embedding_dim (int, optional, defaults to 300) | Dimensionality of embedding layer     |
| hidden_dim (int, optional, defaults to 10)     | Dimensionality of hidden layer        |
| n_layers (int, optional, defaults to 1)        | Number of hidden layers in LSTM       |
| dropout (float, optional, defaults to 0.8)     | Dropout probability for dropout layer |