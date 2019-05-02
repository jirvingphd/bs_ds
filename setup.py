#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# with open('requirements.txt') as req_file:
#     requirements = req_file.read()

setup_requirements = [ ]

test_requirements = [ ]
requirements = [Click>=6.0, numpy>=1.15.4, pandas>=0.24.1, seaborn>=0.9.0, matplotlib>=3.0.2, scikit-learn>=0.0, pydotplus>=2.0.2, scipy>=1.2.1, xgboost>=0.80, IPython>=7.2.0]
setup(
    author="James Irving",
    author_email='james.irving.phd@outlook.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="BroadSteel_DataScience tools.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='bs_ds',
    name='bs_ds',
    packages=find_packages(include=['bs_ds']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jirvingphd/bs_ds',
    version='0.1.1',
    zip_safe=True,
)
