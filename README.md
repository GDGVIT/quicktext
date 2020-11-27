
<p align="center">
<a href="https://dscvit.com">
	<img src="https://user-images.githubusercontent.com/30529572/72455010-fb38d400-37e7-11ea-9c1e-8cdeb5f5906e.png" alt="DSC VIT Logo"/>
</a>
	<h2 align="center"> QuickText </h2>
	<h4 align="center">Toolkit For Text Classification <h4>
</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Install</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#examples">Examples</a>
  <br> <br>

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![DOCS](https://img.shields.io/badge/Docs-latest-green.svg)](https://picturate.github.io/quickTextCassifier/) 
![CI Tests](https://github.com/GDGVIT/quicktext/workflows/CI%20Tests/badge.svg)
[![codecov](https://codecov.io/gh/picturate/qtc/branch/master/graph/badge.svg)](https://codecov.io/gh/GDGVIT/quicktext)

# About

Quicktext is a framework for developing LSTM and CNN based text classification models.

# Features

- It is __easy__ to learn and use quicktext
- The classifiers can be added to __sPacy__ pipeline
- It's built using __PyTorch__, hence has inbuilt __quantization__ and __onnx__ support

# Installation

Install from source

```
pip install -q git+https://github.com/GDGVIT/quicktext.git
```

# Getting Started

```python
from quicktext import TextClassifier
from quicktext import Trainer
from quicktext.datasets import get_imdb

imdb = get_imdb()

classifier = TextClassifier(num_class=2, arch='bilstm')

trainer = Trainer(classifier)
trainer.fit(imdb.train, imdb.val, epochs=10, batch_size=64, gpus=1)
```

# Supported Models


- Bidirectional LSTM
- CNN 2D filters     
- Fasttext           
- RCNN               
- Seq2Seq Attention  


# Examples

- [Spam or ham, spam classification]()

# Contributors

<table>
<tr align="center">


<td>

Ramaneswaran

<p align="center">
<img src = "https://avatars0.githubusercontent.com/u/51799927?s=460&u=3a1e26881d54bc1c4cf2719f976aaa6783db0f54&v=4" width="150" height="150" alt="Raman">
</p>
<p align="center">

<a href = "https://github.com/ramaneswaran"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" alt="profile" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/ramaneswaran-s-76622416b/">

<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36" alt="linkedin"/>
</a>
</p>
</td>
</tr>
</table>

<p align="center">
	Made with :heart: by <a href="https://dscvit.com">DSC VIT</a>
</p>

