import unittest
import torch
from quicktext import TextClassifier


class TextClassifierTester(unittest.TestCase):
    def test_init(self):
        classifier = TextClassifier(num_class=2)

        self.assertTrue(isinstance(classifier, TextClassifier))

    def test_predict(self):
        classifier = TextClassifier(num_class=2)

        text = "Sample text to test the classifier"
        output = classifier.predict(text)
        self.assertIsInstance(output,dict)

    def test_get_ids(self):
        classifier = TextClassifier(num_class=2)

        text = "Sample text to test the classifier"
        ids = classifier.get_ids(text)
        self.assertTrue(isinstance(ids, list))


if __name__ == "__main__":
    unittest.main()
