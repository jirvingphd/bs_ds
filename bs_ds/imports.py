# -*- coding: utf-8 -*-
"""Convience module. 'from bs_ds.imports import *' will pre-load pd,np,plt,mpl,sns"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from IPython.display import display
from IPython.display import HTML
import sklearn

import_dict = {'pandas':'pd',
                'numpy':'np',
                'matplotlib':'mpl',
                'matplotlib.pyplot':'plt',
                'seaborn':'sns'}

df_imported= pd.DataFrame.from_dict(import_dict,orient='index')
df_imported.columns=['Module/Package Handle']
display(df_imported)

