#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# with open('requirements.txt') as req_file:
#     requirements = req_file.read()

setup_requirements = [ ]
version = '0.1.6'
test_requirements = [ ]
requirements = ['Click','numpy', 'pandas', 'seaborn', 'matplotlib', 'scikit-learn', 'pydotplus', 'scipy', 'xgboost' , 'IPython']
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
    version='0.1.6',
    zip_safe=True,
    cmdclass=cmdclass,
        command_options={
        'build_sphinx': {
            'project': ('setup.py', 'bs_ds'),
            'version': ('setup.py', version),
            'release': ('setup.py', ''),
            'source_dir': ('setup.py', 'bs_ds/')}
            }
)