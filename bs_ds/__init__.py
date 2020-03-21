# -*- coding: utf-8 -*-

"""Top-level package for bs_ds."""

__author__ = """James Irving, Michael Moravetz"""
__email__ = 'james.irving.phd@outlook.com'
__version__ = '0.11.0'

from .bs_ds import *
from .bamboo import *
from .prettypandas import *
from .glassboxes import *
from .saywhat import *
from .capstone import * #ihelp, module_menu

# try:
#     import cufflinks as cf
#     cf.go_offline()
#     # '>> `df.iplot()` is enabled.'
#     # print('[i] df.iplot() should be available.')
# except:
#     pass
    



print(f"bs_ds  v{__version__} loaded.  Read the docs: https://bs-ds.readthedocs.io/en/latest/index.html")
print(f"> For convenient loading of standard modules use: `from bs_ds.imports import *`\n")

# def welcome_message():
#     from IPython.display import Markdown as md
#     logo = "<img src='https://raw.githubusercontent.com/jirvingphd/bs_ds/master/docs/bs_ds_logo.png' width=75> \n"
#     msg = f"- For convenient loading of standard modules use:\n```python\nfrom bs_ds.imports import *\n```"
#     msg2 = f"- For a Dropdow Menu of Available Functions' Help and Source Code:\n```python\nbs_bs.module_menu()\n```"
#     return md(logo+msg+'\n'+msg2)
# welcome_message()

# print(f'Modules Displayed in Table Imported to Use\n(Available if used from bs_ds import *)\n')
