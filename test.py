import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from qtc.data import TextClassifierData
from qtc.nets import BiLSTM

from pytorch_lightning import trainer

df_path = "../data/aclImdb/train/data.csv"
df = pd.read_csv(df_path)

texts = df["text"].tolist()
labels = df["label"].tolist()

# data = TextClassifierData(texts=texts, labels=labels)
# loader = DataLoader(data, batch_size=4, collate_fn=data.get_batch)

EMBEDDING_DIM = 300
HIDDEN_DIM = 128
OUTPUT_DIM = 2
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = BiLSTM(EMBEDDING_DIM, OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT,)
