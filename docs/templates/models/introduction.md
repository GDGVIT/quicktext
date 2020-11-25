Quicktext provides several text classification models built using PyTorch

## Benchmark

The table below the shows the model performance on the IMDB movie review dataset and the 20 newsgroups dataset

=== "IMDB Movie Reviews"

    Trained using Nvidia Tesla T4 on Google Colab

    | Model    | Test Accuracy | Epoch Time | Epochs Trained |
    |----------|---------------|------------|----------------|
    | bilstm   | 87.87         | 1:33 mins  | 7              |
    | cnn2d    | 85.33         | 0:52 mins  | 4              |
    | fasttext | 85.50         | 0:35 mins  | 3              |
    | rcnn     | 86.41         | 0:52 mins  | 4              |
    | seq2seq  | 84.01         | 1:22 mins  | 6              |

=== "20 Newsgroups"

    | Model    | Epoch Time | Test F1 score |
    |----------|------------|---------------|
    | bilstm   |            |               |
    | cnn2d    |            |               |
    | fasttext |            |               |
    | rcnn     |            |               |
    | seq2seq  |            |               |


## Using different models

Set the __arch__ argument in TextClassifier class to use different model architectures

```python
# This TextClassifier will use fasttext model
classifier = quicktext.TextClassifier(num_class=20, arch='fasttext')
```
