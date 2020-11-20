In this example we will train a text classification model on the IMDB movie review dataset

```python
from quicktext import TextClassifier
from quicktext import Trainer
from quicktext.datasets import get_imdb

imdb = get_imdb(return_x_y=True)

classifier = TextClassifier(num_class=2)

trainer = Trainer(classifier)
trainer.fit(imdb.train_data, imdb.val_data, epochs=10, batch_size=64, gpus=1)
```