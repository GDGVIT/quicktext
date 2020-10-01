import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from qtc.data import TextClassifierData

df_path = "../data/aclImdb/train/data.csv"
df = pd.read_csv(df_path)

texts = df["text"].tolist()
labels = df["label"].tolist()

data = TextClassifierData(texts=texts, labels=labels)
loader = DataLoader(data, batch_size=4, collate_fn=data.get_batch)

batch = next(iter(loader))

print(batch["texts"].shape)
print(batch["seq_lens"])
