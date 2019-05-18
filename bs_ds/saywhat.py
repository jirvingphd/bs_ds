# -*- coding: utf-8 -*-

"""A collection of language processing tools."""

import nltk
nltk.download('stopwords','punkt')

from nltk.corpus import stopwords
import string
from nltk import word_tokenize, FreqDist
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import fetch_20newsgroups
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
# import pandas as pd
# import numpy as np
# np.random.seed(0)

def make_stopwords(punctuation=True):
    """Makes and returns a stopwords_list for enlgish combined with punctuation(default)."""
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    import string
    stopwords_list = []
    stopwords_list = stopwords.words('english') + list(string.punctuation)
    stopwords_list += ["''", '""', '...', '``']
    return stopwords_list

def process_article(article, stopwords_list=make_stopwords()):
    """Source: Learn.Co Text Classification Lab"""
    import nltk
    tokens = nltk.word_tokenize(article)
    stopwords_removed = [token.lower() for token in tokens if token not in stopwords_list]
    return stopwords_removed
