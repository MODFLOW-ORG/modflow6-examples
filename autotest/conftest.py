from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from modflow_devtools.snapshots import BinaryArrayExtension

pytest_plugins = ["modflow_devtools.fixtures", "modflow_devtools.snapshots"]

EXCLUDE = []
# slowest first, so they don't straggle at the end of parallel runs
SLOW = ["ex-gwf-sagehen", "ex-gwt-rotate", "ex-gwt-uzt-2d", "ex-gwt-hecht-mendez"]
PROJ_ROOT = Path(__file__).parents[1]
SCRIPTS_PATH = PROJ_ROOT / "scripts"
TABLES_PATH = PROJ_ROOT / "tables"
IMAGES_PATH = PROJ_ROOT / "images"
FIGURES_PATH = PROJ_ROOT / "figures"
EXAMPLES_PATH = PROJ_ROOT / "examples"
NOTEBOOKS_PATH = PROJ_ROOT / ".doc" / "_notebooks"
# a rounded snapshot value can flip its last digit when solver roundoff
# changes, so a value may differ from its snapshot by one unit in its last
# rounded decimal (plus float noise); regenerate snapshots at each mf6 release
SNAPSHOT_UNIT_FACTOR = 1.1


def load_csv(path: Path, decimals: int, drop=()) -> np.recarray:
    """Load a CSV output file as a rounded record array."""
    df = pd.read_csv(path).drop(columns=list(drop))
    return df.round(decimals).to_records(index=False)


SNAPSHOT_CONFIG: dict[str, dict[str, partial]] = {
    # TODO: support multiple snapshot files. this is a dictionary to leave the door open for that.
    "ex-gwt-keating": {
        "mf6prt/track.trk.csv": partial(load_csv, decimals=2, drop=["name"]),
    },
    "ex-gwf-maw-collector": {
        "ex-gwf-maw-collector-b/collector.maw.obs.csv": partial(load_csv, decimals=6),
    },
    "ex-gwf-hwell": {
        "ex-gwf-hwell-b/ex-gwf-hwell-b.maw.obs.csv": partial(load_csv, decimals=6),
    },
    "ex-prt-mp7-p01": {
        "prt/mp7-p01-prt.trk.csv": partial(load_csv, decimals=2, drop=["name"]),
    },
    "ex-prt-mp7-p02": {
        "prt/mp7-p02-prt.trk.csv": partial(load_csv, decimals=2, drop=["name"]),
    },
    "ex-prt-mp7-p03": {
        "prt/mp7-p03-prt.trk.csv": partial(load_csv, decimals=2, drop=["name"]),
    },
    "ex-prt-mp7-p04": {
        "prt/mp7-p04-prt.trk.csv": partial(load_csv, decimals=2, drop=["name"]),
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


class TolerantArrayExtension(BinaryArrayExtension):
    """Binary array snapshot that compares numeric fields with a tolerance."""

    atol = 0.0

    def matches(self, *, serialized_data, snapshot_data) -> bool:
        if snapshot_data is None:
            return False
        received = np.load(BytesIO(serialized_data))
        expected = np.load(BytesIO(snapshot_data))
        if received.dtype != expected.dtype or received.shape != expected.shape:
            return False
        for name in received.dtype.names or (None,):
            a = received if name is None else received[name]
            b = expected if name is None else expected[name]
            if np.issubdtype(a.dtype, np.number):
                if not np.allclose(a, b, rtol=0.0, atol=self.atol, equal_nan=True):
                    return False
            elif not np.array_equal(a, b):
                return False
        return True


@pytest.fixture
def snapshot_config(request, example_script, snapshot):
    if request.config.getoption("--snapshot-disable"):
        return None
    example_name = Path(example_script).stem
    config = SNAPSHOT_CONFIG.get(example_name, {})
    if not config:
        return None
    print(f"Snapshot file for {example_name}: {list(config.keys())[0]}")  # noqa: RUF015
    decimals = min(load.keywords["decimals"] for load in config.values())
    extension = type(
        "TolerantArrayExtension",
        (TolerantArrayExtension,),
        {"atol": SNAPSHOT_UNIT_FACTOR * 10.0**-decimals},
    )
    return config, snapshot.use_extension(extension)


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


def pytest_collection_modifyitems(items):
    def rank(item):
        for i, name in enumerate(SLOW):
            if f"[{name}.py]" in item.nodeid:
                return i
        return len(SLOW)

    items.sort(key=rank)
