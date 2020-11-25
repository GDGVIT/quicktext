import unittest
from quicktext.datasets import get_imdb, get_20newsgroups


class TextClassifierTester(unittest.TestCase):
    def test_imdb_download(self):
        imdb = get_imdb()
        self.assertTrue(isinstance(imdb.train.data, list))
        self.assertTrue(isinstance(imdb.train.data[0], str))

    def test_newsgroups_download(self):
        newsgroups = get_20newsgroups()
        self.assertTrue(isinstance(newsgroups.train.data, list))
        self.assertTrue(isinstance(newsgroups.train.data[0], str))


if __name__ == "__main__":
    unittest.main()
