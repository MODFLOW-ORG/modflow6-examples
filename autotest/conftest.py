from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from numpy.typing import NDArray

pytest_plugins = ["modflow_devtools.fixtures", "modflow_devtools.snapshots"]

EXCLUDE = []
PROJ_ROOT = Path(__file__).parents[1]
SCRIPTS_PATH = PROJ_ROOT / "scripts"
TABLES_PATH = PROJ_ROOT / "tables"
IMAGES_PATH = PROJ_ROOT / "images"
FIGURES_PATH = PROJ_ROOT / "figures"
EXAMPLES_PATH = PROJ_ROOT / "examples"
NOTEBOOKS_PATH = PROJ_ROOT / ".doc" / "_notebooks"
SNAPSHOT_CONFIG: dict[str, dict[str, Callable[[Path], NDArray]]] = {
    # TODO: support multiple snapshot files. this is a dictionary to leave the door open for that.
    "ex-gwt-keating": {
        "mf6prt/track.trk.csv": lambda p: pd.read_csv(p)
        .drop("name", axis=1)
        .round(2)
        .to_records(index=False),
    },
}


@pytest.fixture(scope="session")
def write(request) -> bool:
    return not request.config.getoption("--no-write")


@pytest.fixture(scope="session")
def run(request) -> bool:
    return not (
        request.config.getoption("--init") or request.config.getoption("--no-run")
    )


@pytest.fixture(scope="session")
def plot(request) -> bool:
    return request.config.getoption("--plot")


@pytest.fixture(scope="session")
def plot_show(request, plot) -> bool:
    return plot and request.config.getoption("--show")


@pytest.fixture(scope="session")
def plot_save(request, plot) -> bool:
    return plot and not request.config.getoption("--no-save")


@pytest.fixture(scope="session")
def gif(request, plot) -> bool:
    return plot and not request.config.getoption("--no-gif")


@pytest.fixture
def snapshot_config(
    request, example_script, array_snapshot
) -> dict[str, Callable[[Path], NDArray]] | None:
    if request.config.getoption("--snapshot-disable"):
        return None
    example_name = Path(example_script).stem
    config = SNAPSHOT_CONFIG.get(example_name, {})
    if config:
        print(f"Snapshot file for {example_name}: {list(config.keys())[0]}")  # noqa: RUF015
    return (config, array_snapshot) if any(config) else None


def pytest_addoption(parser):
    parser.addoption(
        "--init",
        action="store_true",
        default=False,
        help="Just build and write model input files",
    )
    parser.addoption(
        "--no-write",
        action="store_true",
        default=False,
        help="Disable model build/write",
    )
    parser.addoption(
        "--no-run", action="store_true", default=False, help="Disable model runs"
    )
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="Create plots (disabled by default)",
    )
    parser.addoption(
        "--show",
        action="store_true",
        default=False,
        help="Show plots (disabled by default)",
    )
    parser.addoption(
        "--no-save", action="store_true", default=False, help="Disable plot saving"
    )
    parser.addoption(
        "--no-gif", action="store_true", default=False, help="Disable GIF creation"
    )


def pytest_generate_tests(metafunc):
    # make directories if needed
    TABLES_PATH.mkdir(exist_ok=True)
    IMAGES_PATH.mkdir(exist_ok=True)
    FIGURES_PATH.mkdir(exist_ok=True)
    EXAMPLES_PATH.mkdir(exist_ok=True)
    NOTEBOOKS_PATH.mkdir(exist_ok=True, parents=True)

    # generate example scenarios
    if "example_script" in metafunc.fixturenames:
        scripts = {
            file.name: file
            for file in sorted(SCRIPTS_PATH.glob("ex-*.py"))
            if file.stem not in EXCLUDE
        }
        metafunc.parametrize("example_script", scripts.values(), ids=scripts.keys())
