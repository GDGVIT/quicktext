# Seq2Seq With Attention

## Overview

A sequence to sequence model with attention mechanism for text classification

## Usage

Use __arch='seq2seq'__ while initializing TextClassifier

```python
import quicktext
classifier = quicktext.TextClassifier(num_class=2, arch='seq2seq')
```
## Config

| Parameters                                     | Explanation                           |
|------------------------------------------------|---------------------------------------|
| vocab_size (int, optional, defaults to 26000)  | Vocabulary size the BiLSTM model      |
| embedding_dim (int, optional, defaults to 300) | Dimensionality of embedding layer     |
| hidden_dim (int, optional, defaults to 10)     | Dimensionality of hidden layer        |
| n_layers (int, optional, defaults to 1)        | Number of hidden layers in LSTM       |
| dropout (float, optional, defaults to 0.8)     | Dropout probability for dropout layer |

## References
- [https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch](https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch)