# -*- coding: utf-8 -*-
"""Convience module. 'from bs_ds.importSklearn import  *' will pre-load LogisticRegression, GridSearches, Pipeline
StandardScaler, RobustScaler, MinimaxSCaler, train_test_split, RandomForestClassifer, GradientBoostingClassifier, AdaBoostClassifier"""

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
