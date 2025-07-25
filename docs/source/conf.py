# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

'''
project = 'invode'
copyright = '2025, Raunak Dey'
author = 'Raunak Dey'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']



# conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../invode'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']
html_theme = 'sphinx_rtd_theme'
'''




import os
import sys
sys.path.insert(0, os.path.abspath('../../invode'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

project = 'InvODE'
author = 'Raunak Dey'
release = '0.1'

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax'
    #'sphinx_copybutton',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# If you have Jupyter notebooks with matplotlib plots:
nbsphinx_allow_errors = True  # Optional: allow errors in notebooks to not break build
exclude_patterns = ['_build', '**.ipynb_checkpoints']


templates_path = ['_templates']


html_theme = 'sphinx_book_theme'
html_static_path = ['_static']



