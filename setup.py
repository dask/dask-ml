import sys
import os
from codecs import open

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = ['dask', 'distributed', 'numpy', 'pandas', 'scikit-learn',
                    'scipy', 'dask-glm']

# Optional Requirements
doc_requires = ['sphinx', 'numpydoc', 'sphinx-rtd-theme', 'nbsphinx']
test_requires = ['coverage', 'pytest', 'pytest-mock']
dev_requires = doc_requires + test_requires

if sys.version_info.major == 2:
    test_requires.append("mock")


extra_requires = {
    'docs': doc_requires,
    'test': test_requires,
    'dev': dev_requires,
}

# C Extensions
extensions = [
    Extension(
        "dask_ml.cluster._k_means",
        ["dask_ml/cluster/_k_means.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name='dask_ml',
    description='A library for distributed and parallel machine learning',
    long_description=long_description,
    url='https://github.com/dask/dask-ml',

    author='Tom Augspurger',
    author_email='taugspurger@anaconda.com',
    license='BSD',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Database',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['docs', 'tests']),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=install_requires,
    extras_require=extra_requires,
    ext_modules=cythonize(extensions),
)
