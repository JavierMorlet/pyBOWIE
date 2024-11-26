import os
from setuptools import find_packages, setup

filename = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(filename, 'README.md'), encoding='utf-8') as f:
    README = f.read()

setup(
    name='pybowie',
    version='1.0.5',
    description='Bayesian optimization for constrained or unconstrained, continuous, discrete or mixed data problems',
    author='Javier Morlet-Espinosa, Antonio Flores-Tlacuahuac',
    author_email='a00833961@tec.mx',
    url='https://github.com/JavierMorlet/pyBOWIE',
    long_description=README,
    long_description_content_type="text/markdown",
    packages = find_packages(),
    install_requires=[
        'numpy>=1.23.5',
        'sympy>=1.11.1',
        'pandas>=2.0.3',
        'GPy>=1.10.0',
        'gpflow >= 2.9.2',
        'scipy>=1.10.1',
        'scikit-learn>=1.1.3',
        'properscoring>=0.1',
        'prince>=0.12.1',
        'multiprocess >= 0.70.16',
        'matplotlib>=3.7.3'
        ]
    )