# coding: utf-8# Copyright (c)
# Copyright (c)
# Distributed under the terms of the MIT License.

from setuptools import setup, find_packages

"""setup.py for BayesianChargeAssigner and SparseGroupLasso"""

setup(
    name='high-component-ce-tools',
    version='1.0',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-optimize>=0.8.1', 'scikit-learn>=0.24.1', 'pymatgen>=2020.8.13'],
    url='https://github.com/juliayang/high-component-ce-tools',
    license='MIT',
    author='juliayang',
    author_email='juliayang@berkeley.edu',
    description='tools for building a high-component ce'
)
