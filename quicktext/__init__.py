""" Root package info."""

__version__ = '1.1.0-dev'
__author__ = 'Ramaneswaran S'
__author_email__ = 's.raman2000@outlook.com'
__license__ = 'Apache-2.0'
__copyright__ = "Copyright (c) 2020-2020, %s."%__author__
__homepage__ = 'https://github.com/GDGVIT/quicktext'

__docs__ = (
    "Text Classification models and trainers in PyTorch and sPacy"
)

__long_docs__ = """
    Will write later
"""

from quicktext.classifier.text_classifier import TextClassifier
from quicktext.engine.pl_trainer import Trainer

__all__ = [
    'Trainer',
    'TextClassifier'
]

# for compatibility with namespace packages
__import__('pkg_resources').declare_namespace(__name__)