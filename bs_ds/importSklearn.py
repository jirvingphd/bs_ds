# -*- coding: utf-8 -*-
"""Convience module. 'from bs_ds.importSklearn import  *' will pre-load LogisticRegression, GridSearches, Pipeline
StandardScaler, RobustScaler, Minimax-SCaler, train_test_split, RandomForestClassifer, GradientBoostingClassifier, AdaBoostClassifier.
Also imports basic standard packages from bs_ds.imports"""

def start():
    from bs_ds.bamboo import list2df
    list_sklearn_imports = ['xgboost', 'sklearn', 'scipy', 'sklearn.svm.SVC',
     'sklearn.linear_model.LogisticRegression', 'sklearn.linear_model.LogisticRegressionCV',
      'sklearn.model_selection.RandomizedSearchCV', 'sklearn.model_selection.GridSearchCV',
       'sklearn.pipeline.Pipeline', 'sklearn.decomposition.PCA', 'sklearn.preprocessing.StandardScaler',
       'sklearn.preprocessing.RobustScaler', 'sklearn.preprocessing.MinMaxScaler', 'scipy.stats.randint',
       'scipy.stats.expon', 'sklearn.model_selection.train_test_split', 'sklearn.ensemble.RandomForestClassifier',
        'sklearn.ensemble.GradientBoostingClassifier', 'sklearn.ensemble.AdaBoostClassifier', 'sklearn.tree.DecisionTreeClassifier',
        'sklearn.ensemble.VotingClassifier', 'sklearn.metrics.roc_auc_score']

    list_to_display = [['#','Module']]
    for idx, mod in enumerate(list_sklearn_imports):
        list_to_display.append([idx,mod])

    # from .bamboo import list2df
    df = list2df(list_to_display)
    df.set_index('#',inplace=True)
    display(df)
start()




from .imports import *

import xgboost
import sklearn
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as sts
from IPython.display import display
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.stats import randint, expon
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

import xgboost
import sklearn
import scipy
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.stats import randint, expon
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xbg
from xgboost import XGBClassifier
import time
import re


from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.base import clone
