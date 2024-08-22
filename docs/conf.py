import omlt

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# General information about the project.
project = "omlt"
project_copyright = (
    "2022, Carnegie Mellon University, "
    "Imperial College London, Sandia National Laboratories"
)


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

version = omlt.__version__
release = omlt.__version__

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/omlt_logo.png"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = "omlt-doc"


# -- Options for LaTeX output ------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ("index", "user_guide.tex", "OMLT Documentation", "The OMLT Developers", "manual")
]

# -- External mapping --------------------------------------------------------
intersphinx_mapping = {
    "sphinx": ("http://www.sphinx-doc.org/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "pyscaffold": ("https://pyscaffold.org/en/stable", None),
}
