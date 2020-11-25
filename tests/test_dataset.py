import unittest
from quicktext.datasets import get_imdb, get_20newsgroups


class TextClassifierTester(unittest.TestCase):
    def test_imdb_download(self):
        imdb = get_imdb()
        self.assertIsInstance(imdb.train, list)
        self.assertIsInstance(imdb.train[0][0], str)

    def test_newsgroups_download(self):
        newsgroups = get_20newsgroups()
        self.assertIsInstance(newsgroups.train, list)
        self.assertIsInstance(newsgroups.train[0][0], str)


if __name__ == "__main__":
    unittest.main()
