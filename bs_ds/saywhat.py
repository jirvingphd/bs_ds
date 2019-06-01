# -*- coding: utf-8 -*-

"""A collection of language processing tools."""

# import nltk
# nltk.download('stopwords','punkt')

# from nltk.corpus import stopwords
# import string
# from nltk import word_tokenize, FreqDist
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

class W2vVectorizer(object):
    """From Learn.co Text Classification with Word Embeddings Lab.
    An sklearn-comaptible class containing the vectors for the fit Word2Vec."""

    def __init__(self, w2v, glove):
        # takes in a dictionary of words and vectors as input
        import numpy as np

        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(glove))])

    # Note from Mike: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # It can't be used in a sklearn Pipeline.
    def fit(self, X, y):
        return self

    def transform(self, X):
        import numpy as np
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                   or [np.zeros(self.dimensions)], axis=0) for words in X])
