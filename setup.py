from os.path import exists

from setuptools import setup
import versioneer

install_requires = ["dask >= 0.12.0",
                    "scikit-learn >= 0.18.1",
                    "numpy"]

setup(name='dask-learn',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license='BSD',
      url='http://github.com/dask/dask-learn',
      install_requires=install_requires,
      description='Tools for working with Scikit-Learn and Dask',
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      packages=['dklearn', 'dklearn.tests'])
