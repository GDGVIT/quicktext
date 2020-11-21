# TextClassifier Data

## Overview

This class is a torch Dataset (torch.utils.Dataset)

## Methods

### TextClassifierData.\__init__


| Name  | Type        | Description                          |
|-------|-------------|--------------------------------------|
| data  | list        | List of tuples of form (text, label) |
| vocab | spacy.vocab | Vocabulary for the text classifier   |

### TextClassifierData.collator

This method pads each document (text) in the batch so that dimension of batch is uniform

