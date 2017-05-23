#!/usr/bin/env python

from os.path import exists
from setuptools import setup


setup(name='dask-glm',
      description='Generalized Linear Models with Dask',
      url='http://github.com/dask/dask-glm/',
      maintainer='Matthew Rocklin',
      maintainer_email='mrocklin@gmail.com',
      license='BSD',
      keywords='dask,glm',
      packages=['dask_glm'],
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      install_requires=list(open('requirements.txt').read().strip().split('\n')),
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      extras_require={
          'docs': [
              'jupyter',
              'nbsphinx',
              'notebook',
              'numpydoc',
              'sphinx',
              'sphinx_rtd_theme',
          ]
      },
      zip_safe=False)
