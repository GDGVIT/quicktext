# QuickText

!!! danger "Important"
    This project is still in its early stage of development. Stay Tuned

Hey there, welcome to QuickText

QuickText provides Pytorch text classification models and trainers


## Why QuickText?

Text classification is a simple task. However writing code for a text classifier is not that simple, it takes a lot of time and effort

QuickText aims to save that time and effort by providing 

- Simple API to use a text classifier
- A versatile trainer to train the classifier
- Several model architectures
- Close integration with sPacy, use our classifiers in sPacy pipeline

## Example : Training a sentiment classifier

This code is all it takes to train a sentiment classifier.

```
from quicktext import TextClassifier, Trainer
from quicktext.datasets import get_imdb

imdb = get_imdb(return_x_y=True)

classifier = TextClassifier(num_class=2)

trainer = Trainer(classifier)
trainer.fit(imdb.train_data, imdb.val_data, epochs=10, batch_size=64, gpus=1)
```

<center>
Made with :heart: by <a href="https://dscvit.com">DSC VIT</a>
</center>
