"""Test run example scripts."""

import sys
from os import environ
from pathlib import Path

from modflow_devtools.misc import is_in_ci, run_cmd, set_env


def test_scripts(
    example_script, write, run, plot, plot_show, plot_save, gif, snapshot_config
):
    with set_env(
        WRITE=str(write),
        RUN=str(run),
        PLOT=str(plot),
        PLOT_SHOW=str(plot_show),
        PLOT_SAVE=str(plot_save),
        GIF=str(gif),
    ):
        args = [sys.executable, example_script]
        stdout, stderr, retcode = run_cmd(*args, verbose=True, env=environ)
        assert not retcode, stdout + stderr

    example_name = Path(example_script).stem
    example_workspace = Path(example_script).parent.parent / "examples" / example_name

    # skip snapshots in CI with intel compilers
    skip = is_in_ci() and environ.get("FC", None) in ["ifx", "ifort"]

    if run and snapshot_config and not skip:
        config, snapshot = snapshot_config
        for path, load in config.items():
            if (output_file := example_workspace / path).exists():
                print(f"Comparing snapshot for {output_file}")
                data = load(output_file)
                assert snapshot == data
