# ## Toth Model
#
# A classic conceptual model of groundwater flow in small drainage
# basins is described by Toth (1963).  Using a mathematical model of
# cross-sectional groundwater flow in response to an imposed sinusoidally
# varying water table, Toth postulated the presence of local, intermediate,
# and regional groundwater flow systems.  In his classic paper, Toth
# showed the different types of flow patterns resulting from different
# water table configurations, domain sizes, and aquifer properties.  This
# MODFLOW 6 example is intended to approximate the Toth flow system
# for one of the cases shown in his paper.
#

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
from modflow_devtools.misc import get_env, timed

# Example name and workspace paths. If this example is running
# in the git repository, use the folder structure described in
# the README. Otherwise just use the current working directory.
sim_name = "ex-gwf-toth"
gwf_name = "toth"
try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None
workspace = root / "examples" if root else Path.cwd()
figs_path = root / "figures" if root else Path.cwd()
data_path = root / "data" / sim_name if root else Path.cwd()

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
length_units = "feet"
time_units = "days"

# Model parameters
Lx = 20000.0  # Length in the x direction ($ft$)
top = 10000.0  # Length in the z direction ($ft$)
nper = 1  # Number of periods
nlay = 200  # Number of layers
nrow = 1  # Number of rows
ncol = 100  # Number of columns
bottom = 0.0  # Bottom of model domain ($ft$)
delr = 200.0  # Cell size in x direction ($ft$)
delz = 50.0  # Cell size in z direction ($ft$)
hk = 1.0  # Hydraulic conductivity ($ft/d$)

delr = Lx / ncol
delz = top / nlay
botm = [top - (k + 1) * delz for k in range(nlay)]

# define the Toth parameters
a = 200.0
alpha = np.arctan2(1000, Lx)
period = 5000.0
b = 2 * np.pi / period


# define the water table
def get_z(z0, a, b, alpha, x):
    return z0 + x * np.tan(alpha) + a * np.sin(b * x / np.cos(alpha)) / np.cos(alpha)


x = np.arange(delr / 2, Lx + delr / 2, delr)
z = get_z(top, a, b, alpha, x)

# Time discretization
tdis_ds = ((1.0, 1.0, 1),)

# Well boundary conditions
chdspd = [[0, 0, j, z[j]] for j in range(ncol)]

# Solver parameters
ninner = 100
# -

# ### Model setup
#
# Define functions to build models, write input files, and run the simulation.


# +
def build_models():
    sim_ws = workspace / sim_name
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")

    tdis = flopy.mf6.ModflowTdis(sim)
    ims = flopy.mf6.ModflowIms(sim, print_option="all", inner_maximum=ninner)
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwf_name, save_flows=True)
    dis = flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=nrow, ncol=ncol, top=top, botm=botm, delr=delr
    )
    ic = flopy.mf6.ModflowGwfic(gwf, strt=top)
    npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True, k=hk)
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdspd)
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
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
figure_size = (6, 4)


def plot_results(sim):
    print("Plotting model results...")
    plot_head_results(sim)


def plot_head_results(sim):
    print("Plotting head model results...")
    gwf = sim.gwf[0]

    # load gwf output
    head = gwf.output.head().get_data()
    bud = gwf.output.budget()
    spdis = bud.get_data(text="DATA-SPDIS")[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    # calculate the stream function by summing the flows for each column
    u = qx.reshape((nlay, ncol))
    phi = u[-1::-1].cumsum(axis=0)
    phi = np.flipud(phi)

    with styles.USGSMap():
        fig, ax = plt.subplots(1, 1, figsize=figure_size, dpi=300, tight_layout=True)
        pxs = flopy.plot.PlotCrossSection(model=gwf, ax=ax, line={"row": 0})
        pxs.contour_array(
            head,
            levels=np.arange(top, z.max(), 25),
            linewidths=0.5,
            colors="b",
            linestyles="solid",
        )
        pxs.contour_array(phi, levels=np.linspace(phi.min(), phi.max(), 10))
        ax.plot(x, z, "k-")
        ax.set_xlabel("x position (ft)")
        ax.set_ylabel("elevation (ft)")
        ax.set_aspect(1.0)
        ax.set_xlim(0, 20000)
        ax.set_ylim(0, 11000)

        if plot_show:
            plt.show()
        if plot_save:
            fpth = figs_path / f"{sim_name}.png"
            fig.savefig(fpth)


# -

# ### Running the example
#
# Define a function to run the example scenarios.


# +
def scenario(silent=True):
    sim = build_models()
    if write:
        write_models(sim, silent=silent)
    if run:
        run_models(sim, silent=silent)
    if plot:
        plot_results(sim)


scenario()
# -
