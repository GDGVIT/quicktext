# QuickText

!!! danger "Important"
    This project is still in its early stage of development. Stay Tuned

Hey there, welcome to QuickText

Quicktext is a framework for building and training text classification models

Key features are:

- It is __easy__ to learn and use quicktext
- The classifiers can be added to __sPacy__ pipeline
- It's built using __PyTorch__, hence has inbuilt __quantization__ and __onnx__ support


## Installation

```
pip install -q git+https://github.com/GDGVIT/quicktext.git
```


## Example : Training a sentiment classifier

You can train a movie review sentiment classifier in 2 steps.

### Step 0: Imports

Import following

```python
from quicktext import TextClassifier, Trainer
from quicktext.datasets import get_imdb
```

### Step 1: Dataset and TextClassifier

Prepare the dataset to be used for the model. Quicktext provides two ready to use dataset

We will be using the IMDB movie review dataset

```python
imdb = get_imdb()
```

Next initialize the TextClassifier class
Here you have to pass

- `num_class`: The number of classes in the dataset
- `arch`: The model architecture to use (by default it uses a BiLSTM model)

```python
classifier = TextClassifier(num_class=2, arch='bilstm')
```

### Step 2: Model training

Initialize the trainer to train the classifier

```python
trainer = Trainer(classifier)
trainer.fit(imdb.train, imdb.val, epochs=10, batch_size=64, gpus=1)
```

To the fit method of trainer you need to pass

- `train_data`: A list containing data for model training of form of (text, target) tuples
- `val_data`: A list containing data for model validation
- `epochs`: Number of epochs to train the model 
- `batch_size`: The size of a batch
- `gpus`: Number of gpus to use for training (default set to 0, hence uses CPU)

### Putting it all together

This is the minimal quickstart to build and train a text classication model for movie review sentiment classification

```python
from quicktext import TextClassifier
from quicktext import Trainer
from quicktext.datasets import get_imdb

imdb = get_imdb()

classifier = TextClassifier(num_class=2, arch='bilstm')

trainer = Trainer(classifier)
trainer.fit(imdb.train, imdb.val, epochs=10, batch_size=64, gpus=1)
```

## Further Reading

- [Explore the text classification models in quicktext]()



<center>
Made with :heart: by <a href="https://dscvit.com">DSC VIT</a>
</center>
