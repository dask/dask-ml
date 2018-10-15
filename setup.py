import os
from codecs import open

from setuptools import find_packages, setup

here = os.path.dirname(__file__)


# Get the long description from the README file
with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "dask[array]>=0.18.2",
    "numba",
    "numpy",
    "pandas>=0.21.0",  # CategoricalDtype
    "scikit-learn>=0.20",
    "scipy",
    "dask-glm",
    "six",
    "multipledispatch>=0.4.9",
    "packaging",
]

# Optional Requirements
doc_requires = ["sphinx", "numpydoc", "sphinx-rtd-theme", "nbsphinx", "sphinx-gallery"]
test_requires = [
    "black",
    "coverage",
    "flake8",
    "isort",
    "pytest",
    "pytest-cover",
    "pytest-mock",
]
dev_requires = doc_requires + test_requires
tensorflow_requires = ["dask-tensorflow", "tensorflow"]
xgboost_requires = ["dask-xgboost", "xgboost"]
complete_requires = tensorflow_requires + xgboost_requires

extras_require = {
    "docs": doc_requires,
    "test": test_requires,
    "dev": dev_requires,
    "tensorflow": tensorflow_requires,
    "xgboost": xgboost_requires,
    "complete": complete_requires,
}

setup(
    name="dask-ml",
    description="A library for distributed and parallel machine learning",
    long_description=long_description,
    url="https://github.com/dask/dask-ml",
    author="Tom Augspurger",
    author_email="taugspurger@anaconda.com",
    license="BSD",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=install_requires,
    extras_require=extras_require,
)
