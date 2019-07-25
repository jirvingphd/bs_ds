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
    # nltk.download('stopwords')s
    from nltk.corpus import stopwords
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords',quiet=True)

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


def connect_twitter_api(api_key, api_secret_key):
    """Use tweepy to connect to the twitter-API and return tweepy api object."""
    import tweepy, sys
    auth = tweepy.AppAuthHandler(api_key, api_secret_key)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    if (not api):
        print("Can't authenticate.")
        sys.exit(-1)

    return api


def search_twitter_api(api_object, searchQuery, maxTweets, fName, tweetsPerQry=100, max_id=0, sinceId=None):
    """Take an authenticated tweepy api_object, a search queary, max# of tweets to retreive, a desintation filename.
    Uses tweept.api.search for the searchQuery until maxTweets is reached, saved harvest tweets to fName."""
    import sys, jsonpickle, os, tweepy
    api = api_object
    tweetCount = 0
    print(f'Downloading max{maxTweets} for {searchQuery}...')
    with open(fName, 'a+') as f:
        while tweetCount < maxTweets:

            try:
                if (max_id <=0):
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode='extended')
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, since_id=sinceId, tweet_mode='extended')

                else:
                    if (not sinceId):
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id-1), tweet_mode='extended')
                    else:
                        new_tweets = api.search(q=searchQuery, count=tweetsPerQry, max_id=str(max_id-1),since_id=sinceId, tweet_mode='extended')

                if not new_tweets:
                    print('No more tweets found')
                    break

                for tweet in new_tweets:
                    f.write(jsonpickle.encode(tweet._json, unpicklable=False)+'\n')

                tweetCount+=len(new_tweets)

                print("Downloaded {0} tweets".format(tweetCount))
                max_id = new_tweets[-1].id

            except tweepy.TweepError as e:
                # Just exit if any error
                print("some error : " + str(e))
                break
    print ("Downloaded {0} tweets, Saved to {1}\n".format(tweetCount, fName))

