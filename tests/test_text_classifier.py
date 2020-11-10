import unittest
import en_core_web_md
import torch
from quicktext import TextClassifier


class TextClassifierTester(unittest.TestCase):
    def test_init(self):
        nlp = en_core_web_md.load()
        classifier = TextClassifier(nlp.vocab, n_classes=2)

        self.assertTrue(isinstance(classifier, TextClassifier))

    def test_predict(self):
        nlp = en_core_web_md.load()
        classifier = TextClassifier(nlp.vocab, n_classes=2)

        text = "Sample text to test the classifier"
        output = classifier.predict(text)
        self.assertTrue(isinstance(output.data, torch.Tensor))

    def test_get_ids(self):
        nlp = en_core_web_md.load()
        classifier = TextClassifier(nlp.vocab, n_classes=2)

        text = "Sample text to test the classifier"
        ids = classifier.get_ids(text)
        self.assertTrue(isinstance(ids, list))


if __name__ == "__main__":
    unittest.main()
