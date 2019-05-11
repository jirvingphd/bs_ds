#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'pandas', 'seaborn', 'matplotlib', 'scikit-learn', 'pydotplus', 'scipy', 'xgboost' , 'IPython','pprint'] #Click',

setup_requirements = [ ]

# test_requirements = [ ]
test_requirements = requirements

setup(
    author="James Irving",
    author_email='james.irving.phd@outlook.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',

        'Programming Language :: Python :: 3.5',
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
    packages=find_packages(include=['bs_ds']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jirvingphd/bs_ds',
    version='0.4.6',
    zip_safe=False,
)
