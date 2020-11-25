In this example we will train a text classification model on the 20 newsgroups dataset

## Minimal Quickstart

The following code will build a RCNN text classification model and train it on the 20 newsgroups dataset

```python
from quicktext import TextClassifier
from quicktext import Trainer
from quicktext.datasets import get_20newsgroups

newsgroups = get_20newsgroups(remove=['headers','quotes','footers'])

classifier = TextClassifier(num_class=20, arch='rcnn')

trainer = Trainer(classifier)
trainer.fit(newsgroups.train, newgroups.val, epochs=10, batch_size=64, gpus=1)
```
