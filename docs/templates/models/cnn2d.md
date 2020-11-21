# CNN With 2D Filters

## Overview

This is a CNN model for text classification. It uses 2D filters hence the name CNN2D

## Config

| Parameters                                        | Explanation                                                      |
|---------------------------------------------------|------------------------------------------------------------------|
| vocab_size (int, optional, defaults to 26000)     | Vocabulary size the BiLSTM model                                 |
| embedding_dim (int, optional, defaults to 300)    | Dimensionality of embedding layer                                |
| n_filters (int, optional, defaults to 100)        | Number of convolutional filters applied                          |
| filter_sizes (list, optional, defaults to [3,4,5] | Size of convolution filters                                      |
| dropout (float, optional, defaults to 0.5)        | Randomly drops elements of input tensor with probability dropout |