
<div align="center">
    <img src="images/banner.jpg" width=400 height=100 alt="Banner">
	<h4 align="center"> A Fast & Simple Text Detection Framework  <h4>

</div>

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![DOCS](https://img.shields.io/badge/Docs-latest-green.svg)](https://picturate.github.io/quickTextCassifier/) 


* * * * *
>**Note: This library is still work in progress**
    All contributions are welcome
    Feel free to request any feature or report bugs by creating an issue for it

![CI Tests](https://github.com/picturate/qtc/workflows/CI%20Tests/badge.svg)
![Check Formatting](https://github.com/picturate/qtc/workflows/Check%20Formatting/badge.svg)
![Build mkdocs](https://github.com/picturate/qtc/workflows/Build%20mkdocs/badge.svg)
![Deploy mkdocs](https://github.com/picturate/qtc/workflows/Deploy%20mkdocs/badge.svg)
![PyPi Release](https://github.com/picturate/qtc/workflows/PyPi%20Release/badge.svg)
![Install Package](https://github.com/picturate/qtc/workflows/Install%20Package/badge.svg)
[![codecov](https://codecov.io/gh/picturate/qtc/branch/master/graph/badge.svg)](https://codecov.io/gh/picturate/qtc)
* * * * *

QuickText is a text classification framework with two main features:
- Fast training and inference
- Simple and easy training pipeline 

QuickText is built on top of sPacy and PyTorch and the components provided can be extended and modified if required

- [More about QuickText](#more-about-quicktext)


## Available Models

| Model Class | Name | Docs |
|:-----------:|:---------------------------------------------:|:-------------:|
| BiLSTM | Bidirectional LSTM  | [Click here]() |
| CNN2D| 2D Convolutional Net  | [Click here]() |

> We are currently adding more models to this framework

## More About QuickText

QuickText uses the following libraries

| Library | Used for |
| ---- | --- |
| [**sPacy**](https://spacy.io/) | Text processing and getting feature vectors |
| [**PyTorch**](https://pytorch.org/) | Building neural networks for text classification |
| [**Pytorch_lightning**](https://pytorch-lightning.readthedocs.io/en/stable/) | Model training |

