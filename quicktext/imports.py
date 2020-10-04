# General imports
import os
import random
import logging

# Utility imports
from tqdm import tqdm
import pickle
import gdown
import zipfile

# Machine Learning libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# sPacy imports
import spacy
from spacy.tokenizer import Tokenizer
import en_core_web_md

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
