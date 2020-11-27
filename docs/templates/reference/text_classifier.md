# Text Classifier 



## Overview

The TextClassifier class contains the text classification model and vocabulary.

## Properties

### Model

Initializes a model (torch.nn.Module) for the TextClassifier class

### Vocab

Initialized a vocabulary (spacy.vocab) for the TextClassifier class

## Methods

### TextClassifier.\__init__ 

Initializes the class

| Name                                                     | Type        | Description                                                                                               |
|----------------------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------|
| num_class(mandatory, defaults to 2)                      | int         | Number of classes                                                                                         |
| vocab (optional, defaults to pre-trained en_core_web_md) | spacy.vocab | Spacy vocabulary class                                                                                    |
| arch (optional, defaults to "cnn2d")                      | str         | The architecture of the neural network for text classification                                            |
| config (optional, defaults to None)                      | dict        | A dictionary containing any model parameters, these parameters will override the default model parameters |



### TextClassifier.predict 

Predict the class of the text

```python
classifier = TextClassifier(num_class=2)
scores = classifer.predict(string)
```

| Name  | Type | Description                   |
|-------|------|-------------------------------|
| text  | str  | Input text for classification |

