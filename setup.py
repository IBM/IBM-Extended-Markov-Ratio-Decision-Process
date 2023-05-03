from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'ray[default,serve,k8s]>=1.13.0',
    'ibm-cos-sdk>=2.10.0',
    'boto3>=1.17.110',
    'aiohttp>=3.7.4',
    'aioredis>=1.3.1',
    'scipy'
]

setup(name='emrdp',
version="0.1.0",
description='Extended Markov Ratio Decision Processes.',
long_description_content_type="text/markdown",
long_description=open('README.md').read(),
author='Orit Davidovich, Alexander Zadorojniy',
author_email='orit.davidovich@ibm.com, ZALEX@il.ibm.com',
classifiers=[
    "Programming Language :: Python :: 3",
    'Programming Language :: Python :: 3 :: Only',
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent"
    ],
packages=find_packages(include=['emrdp', 'grid']),
install_requires=install_requires,
package_data={'emrdp': ['notebooks/*.ipynb', 'scripts/*.py']},
python_requires='>=3.8'
)
