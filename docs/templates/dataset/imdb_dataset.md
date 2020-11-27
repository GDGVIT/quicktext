# IMDB Large Movie Review Datset

## Overview 

This is a large dataset for binary sentiment classification. This dataset contains about 25,000 highly polar movie reviews for training and 25,000 more reviews for testing. 

## Usage 

Quicktext has a function to download and parse the large movie reviews dataset. 
The dataset is directly downloaded from Stanford data archives

```python
from quicktext.datasets import get_imdb

imdb = get_imdb()
```

## Dataset structure

The get_imdb function returns a dictionary with the following keys: train,val, test

Each of these keys link to a list of (text, target) tuples.

__Accessing the data__

```python hl_lines="3"
imdb = get_imdb()

imdb.train
```

__Output__
```python
[(sample_text_1..., target_label_1), .... ]
```

