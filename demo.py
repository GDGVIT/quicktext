import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

from quicktext import TextClassifier
from quicktext import Trainer

path = r"../data/aclImdb/train/data.csv"

print("[INFO] Loading spacy model")
nlp = spacy.load("en_core_web_md")

print("[INFO] Loading classifier data")
classifier = TextClassifier(nlp.vocab, n_classes=2)

output = classifier.predict("This is text needs to be big")
print(output.data)

# print("[INFO] Preparing training data")
# df = pd.read_csv(path)
# data = [(row[0], row[1]) for index, row in df.iterrows()]

# train_data, test_data = train_test_split(data, test_size=0.2, random_state=1)

# train_data, val_data = train_test_split(
#     train_data, test_size=0.25, random_state=1
# )  # 0.25 x 0.8 = 0.2


# print("[INFO] Training data")
# trainer = Trainer(classifier, train_data, val_data, test_data, batch_size=2)
# trainer.fit(epochs=1, gpus=0)
