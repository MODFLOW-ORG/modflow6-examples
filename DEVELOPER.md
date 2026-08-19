# Developing MODFLOW 6 examples

This document describes development procedures and conventions for the MODFLOW 6 example models.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Overview](#overview)
- [Requirements](#requirements)
  - [Create a Python environment](#create-a-python-environment)
  - [Install MODFLOW programs](#install-modflow-programs)
  - [Update FloPy classes](#update-flopy-classes)
- [Conventions](#conventions)
  - [Workspace and data access](#workspace-and-data-access)
  - [Structure and configuration](#structure-and-configuration)
- [Running the examples](#running-the-examples)
  - [Running with `pytest`](#running-with-pytest)
  - [Running with `jupytext`](#running-with-jupytext)
  - [Running with `jupyter`](#running-with-jupyter)
- [Building documentation](#building-documentation)
  - [ReadTheDocs](#readthedocs)
  - [PDF document](#pdf-document)
- [Contributing examples](#contributing-examples)
- [Releasing the examples](#releasing-the-examples)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Overview

The example models in this repository are composed with FloPy in the Python files under `scripts/`. Notebooks are generated from these with [`jupytext`](https://github.com/mwouts/jupytext) for the [online documentation](https://modflow6-examples.readthedocs.io/en/stable/) and/or for use with `jupyter`. A small volume of supporting data live in files under `data/`.

## Requirements

To develop the examples one must first:

- create a Python environment
- install MODFLOW programs
- update FloPy classes

### Create a Python environment

Several Python packages are required to run the example scripts. A Conda environment is provided in `<project root/environment.yml`.

### Install MODFLOW programs

Besides MODFLOW 6, the examples use a number of MODFLOW-related programs, discovered on the system `PATH`.

After activating the Conda environment, these can all be installed with:

```shell
get-modflow :
```

You will be prompted to select an install location &mdash; your virtual environment's bindir is a good place, as it is already on the system `PATH` and keeps things isolated.

Typically one develops against the cutting edge of MF6 &mdash; this might entail adding a local development version of MF6 to the bindir as well. Alternatively, the MF6 nightly build can be installed with:

```shell
get-modflow --repo modflow6-nightly-build :
```

See the [FloPy documentation](https://flopy.readthedocs.io/en/stable/md/get_modflow.html) for more info on the `get-modflow` utility.

### Update FloPy classes

FloPy and MODFLOW 6 versions must be kept in sync for FloPy to properly generate and consume MF6 input/output files. To update FloPy from some branch of the [MODFLOW 6 repository](https://github.com/MODFLOW-ORG/modflow6), for instance the `develop` branch:

```shell
python -m flopy.mf6.utils.generate_classes --ref develop
```

The `--owner` and `--repo` arguments can be used to select an alternative repository (e.g. your fork). See [the FloPy documentation](https://flopy.readthedocs.io/en/stable/md/generate_classes.html) for more info.

## Conventions

Example scripts should follow a number of conventions. These keep behavior consistent, reduce barriers to entry for new users, and avoid surprises when building the examples documentation.

### Workspace and data access

Examples may need supporting data which is too large or unwieldy to define in code. For instance, large arrays are better stored in external files, and loaded by the example script at runtime. At the same time, we prefer to avoid implicit dependencies on the local filesystem, so that scripts or notebooks downloaded from the [documentation site](https://modflow6-examples.readthedocs.io/en/stable/) can be run without first cloning the repository and/or manually downloading data files.

To solve this problem, example scripts employ two access patterns:

1. If the script is running inside the repository, the `data/` folder may be assumed to exist, and local files can be accessed directly. Each script creates one or more subdirectories of `examples/` to use as a workspace, one per example scenario. Figures are written to `figures/`. If the script creates any tables, they are written to `tables/`.
2. If the script is not running inside the repository, the current working directory is used as the example workspace. Model input files, output files and figures are all written to that location. Data files are downloaded from GitHub with a tool called [Pooch](https://www.fatiando.org/pooch/latest/). 

A snippet like the following, which determines whether the example is running inside the repository and sets workspace paths, should appear at the beginning of each example script:

```python
sim_name = "ex-gwf-advtidal"
try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None
workspace = root / "examples" if root else Path.cwd()
figs_path = root / "figures" if root else Path.cwd()
data_path = root / "data" / sim_name if root else Path.cwd()
```

**Note:** The build automation expects the simulation name `sim_name` and workspace directory name to match the example name as listed in `doc/body.tex`.

It is also common to define model names up front, e.g.

```python
gwf_name = sim_name + "-gwf"
gwt_name = sim_name + "-gwt"
```

**Note:** MODFLOW 6 restricts model names to 16-characters &mdash; if the example name (i.e. `sim_name`) is long, model names will need abbreviation, e.g.

```python
# shorten model names so they fit in 16-char limit
gwf_name = sim_name.replace("ex-gwf-", "") + "-gwf"
gwt_name = sim_name.replace("ex-gwt-", "") + "-gwt"
```

Provided a file name `fname` and data directory `data_path`, the following snippet suffices to retrieve a data file:

```python
fpath = pooch.retrieve(
    url=f"https://github.com/MODFLOW-ORG/modflow6-examples/raw/develop/data/{sim_name}/{fname}",
    fname=fname,
    path=data_path,
    known_hash="md5:<MD5 hash of the file>",
)
```

Pooch will first check whether the file already exists in the `data_path`, only downloading the file if necessary.

**Note**: When a new example is first added to the repository, the Pooch URL must point to the `develop` branch, as the data file will not be available on the `master` branch until the next release time &mdash; see the [Releasing the examples](#releasing-the-examples) section below. Once an example has reached a stable status, and data files are unlikely to change, the URL may be changed to point to `master`.

**Note**: File hashes are sensitive to newline characters. To avoid disagreements, this repository's `.gitattributes` file overrules Windows' default newline (CR+LF) in favor of POSIX-style newlines (LF) for all text-based files under `data/`.

### Structure and configuration

Scripts may contain one or more example scenarios. Generally, each scenario proceeds through the following steps:

1. Define & build models
2. Write input files
3. Run models
4. Plot results

Scenarios should be wrapped with a function, such that an example script can run multiple scenarios with e.g.

```python
scenario(0)
scenario(1)
...
```

Scenario steps 2-4 listed above can be toggled programmatically via environment variables, all of which accept case-insensitive `true/false` strings:

- `WRITE`: whether to write model input files to subdirectories of `examples/`
- `RUN`: whether to run models
- `PLOT`: whether to create plots of results
- `PLOT_SHOW`: whether to show static plots (disabled by default when run with `pytest`)
- `PLOT_SAVE`: whether to save static plots to `figures/`
- `GIF`: whether to save GIFs to `figures/` (only relevant for a small subset of scripts)

If any environment variable is not found when a script is run directly, the corresponding behavior should be enabled by default. To parse settings from environment variables:

```python
from modflow_devtools.misc import get_env

write = get_env("WRITE", True)
run = get_env("RUN", True)
plot = get_env("PLOT", True)
plot_show = get_env("PLOT_SHOW", True)
plot_save = get_env("PLOT_SAVE", True)
gif_save = get_env("GIF", True)
```

**Note**: When scripts are run via `pytest`, plots are skipped by default, to avoid the need to manually close plot widgets. Pytest CLI flags can be used to control each switch &mdash; see the [Running with `pytest`](#running-with-pytest) section below for details.

Scripts should contain a function wrapping all model runs for each scenario, generally named `run_models()`. This function should be decorated with `@timed`. This measures the example scenario's runtime, providing a rough performance benchmark. For instance:

```python
from modflow_devtools.misc import timed
from pprint import pformat

@timed
def run_models(sim, silent=False):
    success, buff = sim.run_simulation(silent=silent, report=True)
    assert success, pformat(buff)
```

When the scenario is run, a line like the following will be shown:

```
run_models took 74.64 ms
```

## Running the examples

The example scripts can be run directly from the `scripts/` directory, e.g. `python ex-gwf-twri.py`. The environment variables described above can be used to control their behavior. The examples can also be run with `pytest`, converted to notebooks and/or executed with `jupytext`, or run as notebooks with `jupyter`.

**Note**: notebooks are *not* versioned in this repository &mdash; the `scripts/` are the single source of truth.

### Running with `pytest`

The examples can be tested from the `autotest/` directory, either directly as scripts, or as notebooks.

When run via `pytest`, behaviors can be controlled with `pytest` CLI flags:

- `--init`: just build models and write input files, defaults false
- `--no-write`: don't write models, defaults false
- `--no-run`: don't run models, defaults false
- `--plot`: enable plot creation, defaults false
- `--show`: show plots, defaults false
- `--no-save`: don't save static plots, defaults false
- `--no-gif`: don't create/save gifs, defaults false

The last three only apply if `--plot` is enabled. Plotting is disabled by default to avoid having to manually close plot widgets while running tests.

For instance, to create model input files (without running models or plotting results, to save time):

```shell
pytest -v -n auto test_scripts.py --init
```

### Running with `jupytext`

To convert an example script to a notebook manually with `jupytext` (add `--execute` to run the notebook after conversion):

```shell
jupytext --from py --to ipynb scripts/ex-gwf-twri.py -o notebooks/ex-gwf-twri.ipynb
```

### Running with `jupyter`

To start a Jupyter browser interface, run `jupyter notebook` from the `notebooks/` directory after notebooks have been created with `jupytext`.

## Building documentation

This repository includes a ReadTheDocs site which is built automatically in CI, as well as resources for an example models PDF document distributed with each MODFLOW 6 release. Both can be built and checked locally.

### ReadTheDocs

Notebooks must be created and run (including model runs and plots) before building the ReadTheDocs documentation:

```shell
pytest -v -n auto test_notebooks.py --plot
```

This will 

* create and/or update notebooks in `.doc/_notebooks` from example `scripts/*.py`
* create model workspaces in `examples/`, write input files, and run models
* create parameter tables for example descriptions in `tables/`
* create plots and figures from model outputs in `figures/`

Next, build LaTeX and Markdown files:

```shell
python scripts/create-doc-tables.py
```

Next, create ReStructuredText (RST) index files from the contents of `doc/body.tex`:

```shell
python etc/ci_create_examples_rst.py
```

The docs site can now be built from the `.doc/` directory with `make`:

```shell
make clean
make html
```

To host a local development server, switch to the `.doc/_build/html/` directory and run `python -m http.server`.

### PDF document

The PDF examples document shares several initial steps with the ReadTheDocs site. Scripts must first be run (including model runs and plots):

```shell
pytest -v -n auto test_scripts.py --plot
```

The `test_notebooks.py` also suffices, but in this case the conversion from example scripts to notebooks is unnecessary.

Next, build LaTeX and Markdown files:

```shell
python scripts/create-doc-tables.py
```

Next, create ReStructuredText (RST) index files from the contents of `doc/body.tex`:

```shell
python etc/ci_create_examples_rst.py
```

The PDF document can now be built from the `doc/` directory:

```shell
./build-pdf.sh
```

This script runs multiple passes of `pdflatex` and `bibtex` to build the PDF document:

  1. `pdflatex mf6examples.tex`
  2. `bibtex mf6examples`
  3. `pdflatex mf6examples.tex`
  4. `pdflatex mf6examples.tex`

An equivalent batch script `build-pdf.bat` is also available for Windows.

## Contributing examples

To add a new example:

1. Add a new example script in the `scripts/` folder, use existing example scripts as a template. Use the environment variable settings described above to condition whether the script writes model input files, runs models, and plots results.
2. Add a new LaTeX description in the `doc/sections/` folder
3. Ensure the script runs successfully and creates any expected files
4. Build the documentation locally and verify that everything looks good
5. Open a pull request from your fork to the upstream repo's `develop` branch

## Releasing the examples

GitHub Actions automatically creates a new release whenever code is merged into the `master` branch of this repository. To trigger a release, merge (don't squash) a PR from `develop` into `master`.

GitHub Actions will run the following steps:

1. Build MF6 from the target branch.
2. Regenerate flopy modules suitable for this version of MF6.
3. Generate example model input files. (We only distribute input files, not output files.)
4. Build the PDF documentation as described above.
5. Create a draft release and upload the documentation PDF and a zip archive of model input files as assets.

Inspect and publish the release. After publishing, it is necessary to manually trigger a rebuild of the ReadTheDocs site. This can be done by starting the `rtd` workflow from the GitHub Actions web UI, specifying `refs/heads/master` for both the examples and MF6 repos in the dialog. (The workflow should be used from the `develop` branch.)
