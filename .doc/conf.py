# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re

from modflow_devtools.misc import get_env

# -- set boolean indicating if running on readthedocs server -----------------
on_rtd = get_env("READTHEDOCS", False)

# -- get git ref from which we are building the docs -------------------------
ref = get_env("READTHEDOCS_GIT_IDENTIFIER", "master")

# -- setup regular expression for body.tex -----------------------------------
ex_regex = re.compile(r"\\input{sections/(.*?)\}")

# -- parse body.tex for example order ----------------------------------------
pth = os.path.join("..", "doc", "body.tex")
with open(pth) as f:
    lines = f.read()
examples = []
for v in ex_regex.findall(lines):
    if "ex-" in v:
        examples.append(v.replace(".tex", ""))

# -- Build notebook_examples.rst ---------------------------------------------
with open("notebook_examples.rst", "w") as f:
    lines = "Example notebooks\n"
    lines += (len(lines) - 1) * "-" + "\n\n"
    lines += (
        "The Jupyter Notebooks used to create the input files and figures for \n"
        "each of the MODFLOW 6 `examples <examples.html>`_.\n\n"
    )
    f.write(lines)

    lines = ".. nbgallery::\n"
    lines += "   :name: Examples gallery\n\n"
    for base_name in examples:
        lines += f"   _notebooks/{base_name}\n"
    lines += "\n\n"
    f.write(lines)

# -- Build the example restructured text files -------------------------------
if not on_rtd:
    start_dir = os.getcwd()
    pth = os.path.join("..", "etc")
    os.chdir(pth)
    cmd = ("python", "ci_create_examples_rst.py")
    print(" ".join(cmd))
    os.system(" ".join(cmd))
    os.chdir(start_dir)

# -- Project information -----------------------------------------------------

project = "MODFLOW 6 Examples"
copyright = "2020, MODFLOW 6 Development Team"
author = "MODFLOW 6 Development Team"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",  # lowercase didn't work
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "myst_parser",
    "sphinx_markdown_tables",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Settings for GitHub actions integration
if on_rtd:
    extensions.append("rtds_action")
    rtds_action_github_repo = "MODFLOW-ORG/modflow6-examples"
    rtds_action_path = "."
    rtds_action_artifact_prefix = "rtd-files-for-"
    rtds_action_github_token = os.environ.get("GITHUB_TOKEN", None)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
# The encoding of source files.
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["custom.css", "theme_overrides.css"]

html_context = {
    "display_github": False,
    "github_user": "modflow6-examples",
    "github_repo": "modflow6-examples",
    "github_version": "master",
    "doc_path": ".doc",
}

numfig = True
math_numfig = True
numfig_secnum_depth = 1
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
}
math_eqref_format = "{number}"

autosummary_generate = True
numpydoc_show_class_members = False

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "modflow6-examples"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If false, no module index is generated.
html_domain_indices = False

# If false, no index is generated.
html_use_index = False

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

# Output file base name for HTML help builder.
htmlhelp_basename = "mf6exdoc"

# disable automatic notebook execution (built in CI for now)
nbsphinx_execute = "never"

nbsphinx_prolog = (
    r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/MODFLOW-ORG/modflow6-examples/blob/"""
    + ref
    + r"""/scripts/{{ env.docname.split('/')|last|e + '.py' }}">{{ env.docname.split('/')|last|e + '.py' }}</a>.
      It's also available as a <a href="{{ env.docname.split('/')|last|e + '.ipynb' }}" class="reference download internal" download>notebook</a>.
      <script>
        if (document.location.host) {
          let nbviewer_link = document.createElement('a');
          nbviewer_link.setAttribute('href',
            'https://nbviewer.org/url' +
            (window.location.protocol == 'https:' ? 's/' : '/') +
            window.location.host +
            window.location.pathname.slice(0, -4) +
            'ipynb');
          nbviewer_link.innerHTML = 'View in <em>nbviewer</em>';
          nbviewer_link.innerHTML = 'Or view it on <em>nbviewer</em>';
          nbviewer_link.classList.add('reference');
          nbviewer_link.classList.add('external');
          document.currentScript.replaceWith(nbviewer_link, '.');
        }
      </script>
    </div>
"""
)

# Import Matplotlib to avoid this message in notebooks:
# "Matplotlib is building the font cache; this may take a moment."
