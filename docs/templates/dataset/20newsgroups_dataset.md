# The 20 Newsgroups dataset

## Overview

This is a dataset for multiclass classification. This dataset consists of 20000 messages from 20 Usenet newsgroups

## Usage 

Quicktext has a function to download and parse the 20 newsgroups dataset. 
The dataset is directly downloaded from UCI dataset archives

```python
from quicktext.datasets import get_20newsgroups

newsgroups = get_20newsgroups()
```


## Dataset structure

The get_20newsgroups function returns a dictionary with the following keys: train,val, test

Each of these keys link to a list of (text, target) tuples.

__Accessing the data__

```python hl_lines="3"
newsgroups = get_20newsgroups()

newsgroups.train
```

__Output__
```python
[(sample_text_1..., target_label_1), .... ]
```

