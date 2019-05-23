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
# def sidebar():
#     side_jss = function(){let e=document.querySelector(%E2%80%9C.site-sidebar%E2%80%9D),t=document.querySelector(%E2%80%9C.site-main .module%E2%80%9D),l=document.querySelector(%E2%80%9C.site-main%E2%80%9D),s=document.querySelector(%E2%80%9C.site-widget%E2%80%9D);%E2%80%9Cnone%E2%80%9D==e.style.display?(e.style.display=%E2%80%9Cflex%E2%80%9D,t.style.display=%E2%80%9Cblock%E2%80%9D,l.style.right=%E2%80%9C263px%E2%80%9D,s.style.display=%E2%80%9Cblock%E2%80%9D):(e.style.display=%E2%80%9Cnone%E2%80%9D,t.style.display=%E2%80%9Cnone%E2%80%9D,l.style.right=%E2%80%9C0px%E2%80%9D,s.style.display=%E2%80%9Cnone%E2%80%9D)})();
#     # bundle_path = os.path.join(os.path.split(__file__)[0], "resources", "bundle.js")
#     # with io.open(bundle_path, encoding="utf-8") as f:
#     #     bundle_data = f.read()
#     # logo_path = os.path.join(os.path.split(__file__)[0], "resources", "logoSmallGray.png")
#     # with open(logo_path, "rb") as f:
#     #     logo_data = f.read()
#     # logo_data = base64.b64encode(logo_data).decode('utf-8')
#     display(HTML("<script>{side_jss}</script>"))
#         #"<div align='center'><img src='data:image/png;base64,{logo_data}' /></div>".format(logo_data=logo_data) +
#         #"<script>{bundle_data}</script>".format(bundle_data=bundle_data)
#     #))
