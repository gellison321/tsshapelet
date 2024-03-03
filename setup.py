from distutils.core import setup
from setuptools import find_packages

setup(
  name = 'tsshapelet',
  packages=find_packages(),
  version = '2.2.1',
  license='',
  description = 'A timeseries shapelet extraction tool for python.',
  author = 'Grant Ellison',
  author_email = 'gellison321@gmail.com',
  url = 'https://github.com/gellison321/tsshapelet',
  download_url = 'https://github.com/gellison321/tsshapelet/archive/refs/tags/2.2.1.tar.gz',
  keywords = ['timeseries', 'barycenter', 'data science','data analysis'],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python'
  ],
  install_requires=['numpy','scipy', 'numba'],
)