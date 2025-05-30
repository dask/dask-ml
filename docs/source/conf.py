#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dask-ml documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 09:41:40 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import packaging.version

from dask_ml import __version__ as version

# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "numpydoc",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "distributed": ("https://distributed.dask.org/en/latest/", None),
    "dask_glm": ("https://dask-glm.readthedocs.io/en/latest/", None),
}

numpydoc_class_members_toctree = False
autodoc_default_flags = ["members", "inherited-members"]
autosummary_generate = True
templates_path = ["templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "dask-ml"
copyright = "2017, Dask developers"
author = "Dask developers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
# version = ""

release = packaging.version.parse(version).base_version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["build", "**.ipynb_checkpoints"]
exclude_trees = ["_build", "includes"]

# The name of the Pygments (syntax highlighting) style to use.
# Commenting this out for now, if we register dask pygments,
# then eventually this line can be:
# pygments_style = "dask"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------


html_theme = "dask_sphinx_theme"
html_theme_options = {"logo_only": True}
# html_theme_path = [dask_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "dask-mldoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "dask-ml.tex", "dask-ml Documentation", "Dask developers", "manual")
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "dask-ml", "dask-ml Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "dask-ml",
        "dask-ml Documentation",
        author,
        "dask-ml",
        "One line description of project.",
        "Miscellaneous",
    )
]


extlinks = {
    "issue": ("https://github.com/dask/dask-ml/issues/%s", "GH#"),
    "pr": ("https://github.com/dask/dask-ml/pull/%s", "GH#"),
}
