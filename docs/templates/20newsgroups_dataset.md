# The 20 Newsgroups dataset

This is a dataset for multiclass classification. This dataset consists of 20000 messages from 20 Usenet newsgroups

# Usage 

Quicktext has a function to download and parse the 20 newsgroups dataset. 
The dataset is directly downloaded from UCI dataset archives

```python
from quicktext.datasets import get_20newsgroups

newsgroups = get_20newsgroups()
```

# Dataset structure

By default the dataset returned as a dictionary object
