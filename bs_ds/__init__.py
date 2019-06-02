# -*- coding: utf-8 -*-

"""Top-level package for bs_ds."""

__author__ = """James Irving, Michael Moravetz"""
__email__ = 'james.irving.phd@outlook.com'
__version__ = '0.7.3'

from .bs_ds import *
from .bamboo import *
from .prettypandas import *
from .glassboxes import *
from .saycheese import *
from .saywhat import *

print(f'bs_ds v. {__version__} ... read the docs at https://bs-ds.readthedocs.io/en/latest/index.html')
print(f'For convenient loading of standard modules :\n>> from bs_ds.imports import *\n')
# print(f'Modules Displayed in Table Imported to Use\n(Available if used from bs_ds import *)\n')
