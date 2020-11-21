# FastText

## Overview

Proposed by Facebook research, it is one of the fastest text classification models to train and gives results comparable to LSTM and CNN models.

## Config 

| Parameters                                     | Explanation                       |
|------------------------------------------------|-----------------------------------|
| vocab_size (int, optional, defaults to 26000)  | Vocabulary size the BiLSTM model  |
| embedding_dim (int, optional, defaults to 300) | Dimensionality of embedding layer |
| hidden_dim (int, optional, defaults to 10)     | Dimensionality of hidden layer    |