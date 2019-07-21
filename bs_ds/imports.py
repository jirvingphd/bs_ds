# -*- coding: utf-8 -*-
"""Convience module. 'from bs_ds.imports import *' will pre-load pd,np,plt,mpl,sns"""
# import_dict = {'pandas':'pd',
#                 'numpy':'np',
#                 'matplotlib':'mpl',
#                 'matplotlib.pyplot':'plt',
#                 'seaborn':'sns',
#                 'bs_ds':'bs'
#               }
# for package, handle in import_dict.items():
#     exec(f'import {package} as {handle}')

# from IPython.display import display
# # from IPython.display import HTML

# df_imported= pd.DataFrame.from_dict(import_dict,orient='index')
# df_imported.columns=['Module/Package Handle']

# from .prettypandas import html_on, make_CSS,html_off
# display(df_imported)
import pandas as pd
from IPython.display import display

imports_list = [('pandas','pd','High performance data structures and tools'),
                ('numpy','np','scientific computing with Python'),
                ('matplotlib','mpl',"Matplotlib's base OOP module with formatting artists"),
                ('matplotlib.pyplot','plt',"Matplotlib's matlab-like plotting module"),
                ('seaborn','sns',"High-level data visualization library based on matplotlib"),
                ('bs_ds','bs','Custom data science bootcamp student package')]

for package_tuple in imports_list:
    package=package_tuple[0]
    handle=package_tuple[1]
    exec(f'import {package} as {handle}')

df_imported= pd.DataFrame(imports_list,columns=['Package','Handle','Description'])
display(df_imported.sort_values('Package').style.hide_index().set_caption('Loaded Packages and Handles'))

# print('To disable styled DataFrames run html_off() at the bottom of any cell.\n To re-enable use html_on() at the bottom of any cell.')
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
# html_on()
