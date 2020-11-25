# BiDirectional LSTM

## Overview

BiDirectional LSTMs are an extension of the traditional LSTMS. It involves putting two independent RNNs together. This structure allows the networks to have both backward and forward information about the sequence at every time step

## Usage

Use __arch='bilstm'__ while initializing TextClassifier

```python
import quicktext
classifier = quicktext.TextClassifier(num_class=2, arch='bilstm')
```

## Config

| Parameters                                       | Explanation                                                      |
|--------------------------------------------------|------------------------------------------------------------------|
| vocab_size (int, optional, defaults to 26000)    | Vocabulary size the BiLSTM model                                 |
| embedding_dim (int, optional, defaults to 300)   | Dimensionality of embedding layer                                |
| hidden_dim (int, optional, defaults to 128)      | Dimensionality of hidden layer in LSTM                           |
| n_layers (int, optional, defaults to 2)          | Number of stacked layers in LSTM                                 |
| bidirectional (bool, optional, defaults to True) | The LSTM is bidirectional if True                                |
| dropout (float, optional, defaults to 0.5)       | Randomly drops elements of input tensor with probability dropout |

## References
- [https://github.com/bentrevett/pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)
