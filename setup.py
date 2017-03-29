from os.path import exists

from setuptools import setup
import versioneer

install_requires = ["dask >= 0.12.0",
                    "scikit-learn >= 0.18.0",
                    "numpy"]

setup(name='dask-searchcv',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license='BSD',
      url='http://github.com/dask/dask-searchcv',
      install_requires=install_requires,
      description='Tools for doing hyperparameter search Scikit-Learn and Dask',
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      packages=['dask_searchcv', 'dask_searchcv.tests'])
