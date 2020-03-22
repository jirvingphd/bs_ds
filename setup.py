#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['IPython','pprint', 'seaborn','tzlocal','cufflinks','scikit-learn>=0.22.2','matplotlib>=3.2',
'qgrid','fake-useragent','gensim','openpyxl','beautifulsoup4',#'plotly==3.10.0',
'xgboost','pyperclip','tweepy']  #['numpy', 'pandas', 'seaborn', 'matplotlib', 'scikit-learn', 'pydotplus',
# 'scipy', 'shap','LIME','IPython','pprint','graphviz','nltk','lime', 'cufflinks==0.16','plotly==3.10.0',
# 'qgrid','fake-useragent','keras>=2.2.4','tensorflow','eli5','ipywidgets',
# 'pytz','tzlocal','gensim','openpyxl','beautifulsoup4',
# 'imgkit','xgboost','pyperclip'] #Click'
setup_requirements = requirements

test_requirements_to_add = ['tweepy','jsonpickle','jinja2','catboost','seaborn','nltk'] #,'cufflinks==0.16','plotly==3.10.0']
test_requirements = requirements
[test_requirements.append(x) for x in test_requirements_to_add]
# test_requirements = ['numpy', 'pandas', 'seaborn', 'matplotlib', 'scikit-learn', 'pydotplus', 'scipy', 'shap','IPython','pprint','nltk','lime','catboost','keras'] #,'xgboost']

setup(
    author="James Irving",
    author_email='james.irving.phd@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="A collection of tools from bootcamp.",
    entry_points={
        'console_scripts': [
            'bs_ds=bs_ds.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='bs_ds',
    name='bs_ds',
    packages=find_packages(include=['bs_ds'],exclude=['fakebrain','importSklearn', 'saycheese','waldos_work']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jirvingphd/bs_ds',
    version='0.11.0',
    zip_safe=False,
)
