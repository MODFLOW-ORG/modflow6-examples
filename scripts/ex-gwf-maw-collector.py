# ## Radial Collector Well
#
# A radial collector well, also called a Ranney well, is a central caisson with
# horizontal laterals that radiate outward near the base of an aquifer. This
# example represents a radial collector well as a single multi-aquifer well with
# one vertical connection to the caisson cell and one horizontal connection to
# each cell a lateral passes through. A steady-state simulation shows the head
# field around the well, and a transient simulation of pumping and recovery
# compares the well head with the laterals represented as horizontal
# connections and as thin vertical screens.

# ### Initial setup
#
# Import dependencies, define the example name and workspace, and read settings from environment variables.

# +
from pathlib import Path

import flopy
import git
import matplotlib.pyplot as plt
import numpy as np
from flopy.plot.styles import styles
from matplotlib.lines import Line2D
from modflow_devtools.misc import get_env, timed

# Example name and workspace paths. If this example is running
# in the git repository, use the folder structure described in
# the README. Otherwise just use the current working directory.
sim_name = "ex-gwf-maw-collector"
try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None
workspace = root / "examples" if root else Path.cwd()
figs_path = root / "figures" if root else Path.cwd()

# Settings from environment variables
write = get_env("WRITE", True)
run = get_env("RUN", True)
plot = get_env("PLOT", True)
plot_show = get_env("PLOT_SHOW", True)
plot_save = get_env("PLOT_SAVE", True)
# -

# ### Define parameters
#
# Define model units, parameters and other settings.

# +
# Model units
length_units = "meters"
time_units = "days"

# Scenario-specific parameters. Scenario a is steady state; scenarios b and c
# are transient, with the laterals represented as horizontal connections and as
# thin vertical screens.
parameters = {
    "ex-gwf-maw-collector-a": {"transient": False, "horizontal": True},
    "ex-gwf-maw-collector-b": {"transient": True, "horizontal": True},
    "ex-gwf-maw-collector-c": {"transient": True, "horizontal": False},
}

# Model parameters
nlay = 1  # Number of layers
nrow = 41  # Number of rows
ncol = 41  # Number of columns
delr = 25.0  # Column width ($m$)
delc = 25.0  # Row width ($m$)
top = 50.0  # Top of the model ($m$)
botm = 0.0  # Bottom of the model ($m$)
strt = 50.0  # Starting and constant head ($m$)
k11 = 25.0  # Horizontal hydraulic conductivity ($m/d$)
k33 = 25.0  # Vertical hydraulic conductivity ($m/d$)
ss = 1.0e-4  # Specific storage ($1/m$)
well_radius = 0.5  # Well radius ($m$)
skin_radius = 1.0  # Radius to the outside of the filter pack ($m$)
k_skin = 25.0  # Filter pack hydraulic conductivity ($m/d$)
lateral_cells = 12  # Number of cells each lateral extends from the caisson
steady_rate = 25000.0  # Steady-state pumping rate ($m^3/d$)
transient_rate = 40000.0  # Transient pumping rate ($m^3/d$)
perlen = 5.0  # Length of the pumping and recovery periods ($d$)
nstp = 25  # Number of time steps in each period
tsmult = 1.2  # Time step multiplier

# Static temporal data used by TDIS file
tdis_ds = ((perlen, nstp, tsmult), (perlen, nstp, tsmult))

# The caisson is in the center cell. Each lateral is a horizontal borehole at
# mid-depth, so its connection is screened over one well diameter.
caisson = (0, nrow // 2, ncol // 2)

# Every scenario is in its own folder, so the flow model has one name
gwf_name = "collector"
zwell = 0.5 * (top + botm)
screen_top = zwell + well_radius
screen_bot = zwell - well_radius

# Solver parameters
nouter = 100
ninner = 200
hclose = 1e-9
rclose = 1e-9
# -

# ### Model setup
#
# Define functions to build models, write input files, and run the simulation.


# +
def lateral_cellids():
    # cells traversed by the four laterals, outward from the caisson
    k, ic, jc = caisson
    cellids = []
    for d in range(1, lateral_cells + 1):
        cellids.append((k, ic - d, jc))
        cellids.append((k, ic + d, jc))
        cellids.append((k, ic, jc - d))
        cellids.append((k, ic, jc + d))
    return cellids


def build_models(name, transient=True, horizontal=True):
    sim_ws = workspace / sim_name / name
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=sim_ws, exe_name="mf6")
    if transient:
        flopy.mf6.ModflowTdis(
            sim, nper=len(tdis_ds), perioddata=tdis_ds, time_units=time_units
        )
    else:
        flopy.mf6.ModflowTdis(sim, nper=1, time_units=time_units)
    flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        outer_maximum=nouter,
        outer_dvclose=hclose,
        inner_maximum=ninner,
        inner_dvclose=rclose,
        linear_acceleration="bicgstab",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwf_name, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k11, k33=k33, save_flows=True)
    flopy.mf6.ModflowGwfic(gwf, strt=strt)
    if transient:
        flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=ss, sy=0.0, transient={0: True})

    # constant head around the edge of the domain
    chdspd = []
    for i in range(nrow):
        for j in range(ncol):
            if i in (0, nrow - 1) or j in (0, ncol - 1):
                chdspd.append([(0, i, j), strt])
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd, pname="CHD")

    # one well head for the caisson and the laterals. The caisson connection is
    # vertical and spans the layer. Each lateral cell is a separate connection,
    # horizontal (90 degrees) and one cell long, when the length correction is
    # applied, or a vertical screen one well diameter long when it is not.
    laterals = lateral_cellids()
    connectiondata = [[0, 0, caisson, top, botm, k_skin, skin_radius]]
    angledata = []
    for icon, cellid in enumerate(laterals, start=1):
        connectiondata.append(
            [0, icon, cellid, screen_top, screen_bot, k_skin, skin_radius]
        )
        angledata.append([0, icon, 90.0, delr])
    rate = transient_rate if transient else steady_rate
    maw_kwargs = {
        "save_flows": True,
        "print_head": True,
        "packagedata": [[0, well_radius, botm, strt, "MEAN", len(connectiondata)]],
        "connectiondata": connectiondata,
        "pname": "MAW",
    }
    if transient:
        maw_kwargs["no_well_storage"] = True
        maw_kwargs["perioddata"] = {0: [[0, "rate", -rate]], 1: [[0, "rate", 0.0]]}
        maw_kwargs["observations"] = {f"{gwf_name}.maw.obs.csv": [("head", "head", 1)]}
    else:
        maw_kwargs["perioddata"] = {0: [[0, "rate", -rate]]}
    if horizontal:
        maw_kwargs["non_vertical_wells"] = True
        maw_kwargs["angledata"] = angledata
    flopy.mf6.ModflowGwfmaw(gwf, **maw_kwargs)

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{gwf_name}.hds",
        budget_filerecord=f"{gwf_name}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    return sim


def write_models(sim, silent=True):
    sim.write_simulation(silent=silent)


@timed
def run_models(sim, silent=True):
    success, buff = sim.run_simulation(silent=silent)
    assert success, buff


# -

# ### Plotting results
#
# Define functions to plot model results.


# +
# Figure properties
figure_size_map = (5.0, 5.4)
figure_size_ts = (5.0, 3.5)


def plot_head_map(silent=True):
    name = next(iter(parameters))
    sim_ws = workspace / sim_name / name
    sim = flopy.mf6.MFSimulation.load(sim_ws=sim_ws, verbosity_level=0)
    gwf = sim.get_model(gwf_name)
    head = gwf.output.head().get_data()
    hmin = float(np.nanmin(head))

    with styles.USGSMap():
        fig, ax = plt.subplots(figsize=figure_size_map, layout="constrained")
        pmv = flopy.plot.PlotMapView(model=gwf, ax=ax)
        cb = pmv.plot_array(head, cmap="viridis_r", alpha=0.85)
        cs = pmv.contour_array(
            head, levels=np.linspace(hmin, strt, 9), colors="white", linewidths=0.75
        )
        ax.clabel(cs, fmt="%.1f", fontsize=7)
        pmv.plot_grid(lw=0.2, color="0.7")
        pmv.plot_bc("CHD", color="0.4")
        pmv.plot_bc("MAW", color="red")
        xc = gwf.modelgrid.xcellcenters[caisson[1:]]
        yc = gwf.modelgrid.ycellcenters[caisson[1:]]
        ax.plot(xc, yc, marker="o", mfc="white", mec="black", ms=6, zorder=5)
        ax.set_aspect("equal", "box")
        styles.xlabel(ax=ax, label="x, in meters")
        styles.ylabel(ax=ax, label="y, in meters")
        cbar = fig.colorbar(cb, ax=ax, shrink=0.7)
        cbar.set_label("Head, in meters")
        handles = [
            Line2D([], [], color="red", lw=6, label="Lateral cells"),
            Line2D([], [], color="0.4", lw=6, label="Constant-head cells"),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                mfc="white",
                ls="",
                ms=6,
                label="Caisson",
            ),
        ]
        styles.graph_legend(
            ax=ax,
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            fontsize=7,
        )

        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig(figs_path / f"{sim_name}-map.png", dpi=300)


def read_well_head(idx):
    name = list(parameters.keys())[idx]
    fpth = workspace / sim_name / name / f"{gwf_name}.maw.obs.csv"
    obs = np.genfromtxt(fpth, delimiter=",", names=True)
    return obs["time"], obs["HEAD"]


def plot_well_head(silent=True):
    totim, head_h = read_well_head(1)
    _, head_v = read_well_head(2)

    with styles.USGSPlot():
        fig, ax = plt.subplots(figsize=figure_size_ts, layout="constrained")
        # the pumping period is shaded red and the recovery period blue
        ax.axvspan(0.0, perlen, color="red", alpha=0.2, lw=0.0, zorder=0)
        ax.axvspan(perlen, totim[-1], color="blue", alpha=0.2, lw=0.0, zorder=0)
        ax.plot(totim, head_h, color="black", lw=1.25, label="Horizontal laterals")
        ax.plot(
            totim,
            head_v,
            color="black",
            lw=1.25,
            ls="--",
            label="Thin vertical screens",
        )
        ax.set_xlim(0.0, totim[-1])
        styles.xlabel(ax=ax, label="Time, in days")
        styles.ylabel(ax=ax, label="Head in the well, in meters")
        styles.add_text(ax=ax, text="Pumping", x=0.02, y=0.96, va="top", bold=False)
        styles.add_text(
            ax=ax, text="Recovery", x=0.98, y=0.04, ha="right", va="bottom", bold=False
        )
        styles.graph_legend(ax=ax, loc="center right", fontsize=7)
        styles.remove_edge_ticks(ax=ax)

        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig(figs_path / f"{sim_name}-head.png", dpi=300)


def plot_results(silent=True):
    if not plot:
        return
    plot_head_map(silent=silent)
    plot_well_head(silent=silent)


# -

# ### Running the example
#
# Define and invoke a function to run the example scenario, then plot results.


# +
def scenario(idx=0, silent=True):
    key = list(parameters.keys())[idx]
    params = parameters[key].copy()
    sim = build_models(key, **params)
    if write:
        write_models(sim, silent=silent)
    if run:
        run_models(sim, silent=silent)


# -

# Run the steady-state simulation.

scenario(0)

# Run the transient simulation with the laterals as horizontal connections.

scenario(1)

# Run the transient simulation with the laterals as thin vertical screens.

scenario(2)

# Plot the results.

if plot:
    plot_results()
