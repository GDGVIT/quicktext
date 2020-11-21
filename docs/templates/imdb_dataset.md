# IMDB Large Movie Review Datset

This is a large dataset for binary sentiment classification. This dataset contains about 25,000 highly polar movie reviews for training and 25,000 more reviews for testing. 

# Usage 

Quicktext has a function to download and parse the large movie reviews dataset. 
The dataset is directly downloaded from Stanford data archives

```python
from quicktext.datasets import get_imdb

imdb = get_imdb()
```

# Dataset structure

By default the dataset returned as a dictionary object
