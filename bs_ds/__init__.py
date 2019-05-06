# -*- coding: utf-8 -*-

"""Top-level package for bs_ds."""

__author__ = """James Irving, Michael Moravetz"""
__email__ = 'james.irving.phd@outlook.com'
__version__ = '0.3.3'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as sts
from IPython.display import display
import xgboost
import sklearn
import scipy
from .bs_ds import *
from .bamboo import *
print(f'View our documentation at https://bs-ds.readthedocs.io/en/latest/bs_ds.html')
print(f'Recommended import method:\n>> from bs_ds import *\n')
print(f'Modules Displayed in Table Imported to Use\n(Available if used from bs_ds import *)\n')

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

