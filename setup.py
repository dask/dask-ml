from setuptools import setup

install_requires = ["dask >= 0.12.0",
                    "scikit-learn >= 0.18.1",
                    "numpy"]

setup(name='dask-learn',
      version='0.1.0',
      license='BSD',
      description='Machine Learning with Dask',
      url='http://github.com/dask/dask-learn',
      install_requires=install_requires,
      packages=['dklearn', 'dklearn.tests'])
