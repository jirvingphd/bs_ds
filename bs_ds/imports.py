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

from .prettypandas import html_on, make_CSS,html_off
html_on()
df_imported

print('To disable styled DataFrames use html_off().\n To re-enable use html_on().')

