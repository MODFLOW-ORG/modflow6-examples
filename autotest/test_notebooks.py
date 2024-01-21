"""Convert example scripts to notebooks and test notebook execution."""

from os import environ

from modflow_devtools.markers import requires_exe
from modflow_devtools.misc import run_cmd, set_env

from conftest import NOTEBOOKS_PATH


@requires_exe("jupytext")
def test_notebooks(example_script, write, run, plot, plot_show, plot_save, gif):
    with set_env(
        WRITE=str(write),
        RUN=str(run),
        PLOT=str(plot),
        PLOT_SHOW=str(plot_show),
        PLOT_SAVE=str(plot_save),
        GIF=str(gif),
    ):
        args = [
            "jupytext",
            "--from",
            "py",
            "--to",
            "ipynb",
            "--execute",
            example_script,
            "-o",
            NOTEBOOKS_PATH / example_script.with_suffix(".ipynb").name,
        ]
        stdout, stderr, retcode = run_cmd(*args, verbose=True, env=environ)
        assert not retcode, stdout + stderr
