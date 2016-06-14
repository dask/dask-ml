from setuptools import setup

install_requires = ["dask >= 0.8.2",
                    "scikit-learn >= 0.17.1",
                    "numpy"]

setup(name='dask-learn',
      version='0.1.0',
      license='BSD',
      description='Machine Learning with Dask',
      url='http://github.com/jcrist/dask-learn',
      install_requires=install_requires,
      packages=['dklearn'])
