# ## Horizontal Well in an Anisotropic Confined Aquifer
#
# A horizontal well is screened along a line rather than at a point, so the
# drawdown it produces cannot be described by a radial solution. This example
# reproduces the horizontal well of Zhan and others (2001) and compares three
# ways of representing the well: a specified flux distributed along the screen,
# a multi-aquifer well with non-vertical connections, and, for comparison, the
# Connected Linear Network Process of MODFLOW-USG.

# ### Initial setup
#
# Import dependencies, define the example name and workspace, and read settings from environment variables.

# +
import argparse
from pathlib import Path

import flopy
import git
import matplotlib.pyplot as plt
import numpy as np
from flopy.plot.styles import styles
from matplotlib.lines import Line2D
from modflow_devtools.misc import get_env, timed
from scipy.integrate import quad
from scipy.special import erf

# Example name and workspace paths. If this example is running
# in the git repository, use the folder structure described in
# the README. Otherwise just use the current working directory.
sim_name = "ex-gwf-hwell"
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

# The MODFLOW-USG results are read from a file so that the example runs without
# MODFLOW-USG. Pass --mfusg, or set the MFUSG environment variable, to build and
# run the MODFLOW-USG models and rewrite that file.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--mfusg", action="store_true")
build_usg = _parser.parse_known_args()[0].mfusg or get_env("MFUSG", False)
# -

# ### Define parameters
#
# Define model units, parameters and other settings.

# +
# Model units
length_units = "meters"
time_units = "days"

# Scenario-specific parameters
parameters = {
    "ex-gwf-hwell-a": {"well": "WEL"},
    "ex-gwf-hwell-b": {"well": "MAW"},
}

# Model parameters
nper = 1  # Number of periods
nlay = 11  # Number of layers
nrow = 118  # Number of rows
ncol = 118  # Number of columns
delr_min = 1.25  # Minimum column width ($m$)
delc_min = 1.25  # Minimum row width ($m$)
top = 10.0  # Top of the model ($m$)
thickness = 10.0  # Aquifer thickness ($m$)
strt = 50.0  # Starting head ($m$)
k11 = 10.0  # Horizontal hydraulic conductivity ($m/d$)
k33 = 1.0  # Vertical hydraulic conductivity ($m/d$)
ss = 1.0e-4  # Specific storage ($1/m$)
screen_length = 100.0  # Length of the well screen ($m$)
screen_elevation = 5.0  # Elevation of the well screen ($m$)
well_radius = 0.1  # Well radius ($m$)
skin_radius = 0.15  # Radius to the outside of the filter pack ($m$)
k_skin = 10.0  # Filter pack hydraulic conductivity ($m/d$)
pumping_rate = 1000.0  # Pumping rate ($m^3/d$)
perlen = 1.0  # Length of the stress period ($d$)
nstp = 60  # Number of time steps
tsmult = 1.2  # Time step multiplier

# Static temporal data used by TDIS file
tdis_ds = ((perlen, nstp, tsmult),)

# Layer bottoms. An odd number of layers puts a layer center on the well.
delz = thickness / nlay
botm = top - delz * np.arange(1, nlay + 1)
zcenters = botm + 0.5 * delz
kwell = int(np.argmin(np.abs(zcenters - screen_elevation)))


def grid_spacing(fine, n_fine, factor, edge, cap=200.0):
    """Return cell widths: uniform near the well, expanding to the edge."""
    widths = [fine] * n_fine
    while sum(widths) < edge:
        widths.append(min(widths[-1] * factor, cap))
    return np.array(widths)


# Row and column widths, symmetric about the center of the screen
half = grid_spacing(delr_min, 48, 1.7, 800.0)
delr = np.concatenate((half[::-1], half))
delc = np.concatenate((half[::-1], half))
xedges = np.concatenate(([0.0], np.cumsum(delr)))
yedges = np.concatenate(([0.0], np.cumsum(delc)))
xcenters = 0.5 * (xedges[:-1] + xedges[1:]) - 0.5 * xedges[-1]
ycenters = 0.5 * (yedges[:-1] + yedges[1:]) - 0.5 * yedges[-1]
xedges = xedges - 0.5 * xedges[-1]
yedges = yedges - 0.5 * yedges[-1]

# Cells penetrated by the screen, and the row holding it
jwell = np.where(np.abs(xcenters) <= 0.5 * screen_length)[0]
iwell = int(np.argmin(np.abs(ycenters)))
# y of each model row; row 0 is at the largest y, so the row index has to be
# read through this array rather than through ycenters
yrows = ycenters[::-1]
# the grid is symmetric about the well, so no row is centered on it; the
# analytical solution is evaluated relative to the row the well occupies
ywell = yrows[iwell]

# Observation points, at cell centers
observations = (
    (0.0, 5.0, zcenters[kwell], "$x =$ 0, $y =$ 5 m"),
    (0.0, 20.0, zcenters[kwell], "$x =$ 0, $y =$ 20 m"),
    (75.0, 0.0, zcenters[kwell], "$x =$ 75 m, $y =$ 0"),
)

# Window and times used for the difference maps
map_extent = (110.0, 70.0)
map_times = (0.1, 1.0)
map_cols = np.where(np.abs(xcenters) <= map_extent[0])[0]
map_rows = np.where(np.abs(yrows - ywell) <= map_extent[1])[0]

# Solver parameters
nouter = 100
ninner = 200
hclose = 1e-8
rclose = 1e-9
# -

# ### Analytical solution
#
# Zhan and others (2001) give the drawdown produced by a horizontal well of
# finite length in an anisotropic confined aquifer bounded above and below by
# no-flow boundaries. The well is a uniform-flux line sink, so the solution
# assumes that discharge is distributed evenly along the screen.


# +
def _vertical_series(t, zd, zwd, nterms=200):
    """Sum the vertical eigenfunction series for the no-flow aquifer."""
    n = np.arange(1, nterms + 1)
    return 1.0 + 2.0 * np.sum(
        np.cos(n * np.pi * zd)
        * np.cos(n * np.pi * zwd)
        * np.exp(-(n**2) * np.pi**2 * t)
    )


def drawdown_analytical(x, y, z, t):
    """Return the drawdown of Zhan and others (2001), equation 23."""
    a = np.sqrt(k33 / k11)
    xd, yd, zd = x / thickness * a, y / thickness * a, z / thickness
    ld, zwd = screen_length / thickness * a, screen_elevation / thickness
    td = k33 * t / (ss * thickness**2)

    def integrand(s):
        if s <= 0.0:
            return 0.0
        r = 2.0 * np.sqrt(s)
        e = erf((0.5 * ld + xd) / r) + erf((0.5 * ld - xd) / r)
        return (
            e / np.sqrt(s) * np.exp(-(yd**2) / (4.0 * s)) * _vertical_series(s, zd, zwd)
        )

    value, _ = quad(integrand, 0.0, td, limit=400, epsabs=1e-10, epsrel=1e-10)
    sd = np.sqrt(np.pi) / (2.0 * ld) * value
    return sd * pumping_rate / (2.0 * np.pi * k11 * thickness)


# -

# ### Model setup
#
# Define functions to build models, write input files, and run the simulation.


# +
def build_models(name, well="WEL"):
    sim_ws = workspace / sim_name / name
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=sim_ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)
    flopy.mf6.ModflowIms(
        sim,
        print_option="summary",
        outer_maximum=nouter,
        outer_dvclose=hclose,
        inner_maximum=ninner,
        inner_dvclose=rclose,
        linear_acceleration="bicgstab",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
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
        xorigin=xedges[0],
        yorigin=yedges[0],
    )
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k11, k33=k33)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=ss, sy=0.0, transient={0: True})
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    if well == "WEL":
        # discharge divided evenly among the cells the screen passes through,
        # which is the distribution the analytical solution assumes
        rate = -pumping_rate / jwell.size
        flopy.mf6.ModflowGwfwel(
            gwf,
            stress_period_data=[[(kwell, iwell, j), rate] for j in jwell],
        )
    else:
        # one well head over the whole screen. A connection that passes through
        # more than one cell must be split, so the screen is a separate
        # connection to each cell, each horizontal and one cell long. The MEAN
        # conductance equation is used because the radial equations assume
        # convergence in the horizontal plane.
        connectiondata = [
            [
                0,
                i,
                (kwell, iwell, j),
                screen_elevation + well_radius,
                screen_elevation - well_radius,
                k_skin,
                skin_radius,
            ]
            for i, j in enumerate(jwell)
        ]
        angledata = [[0, i, 90.0, delr[j]] for i, j in enumerate(jwell)]
        maw = flopy.mf6.ModflowGwfmaw(
            gwf,
            # the analytical solution has no wellbore storage
            no_well_storage=True,
            non_vertical_wells=True,
            nmawwells=1,
            packagedata=[
                [0, well_radius, botm[kwell], strt, "MEAN", len(connectiondata)]
            ],
            connectiondata=connectiondata,
            angledata=angledata,
            perioddata={0: [[0, "rate", -pumping_rate]]},
            head_filerecord=f"{name}.maw.hds",
        )
        # well head and the discharge of connections spanning the screen, which
        # is what the non-vertical connections determine
        obs_file = f"{name}.maw.obs"
        maw.obs.initialize(
            filename=obs_file,
            digits=10,
            continuous={
                obs_file + ".csv": [("head", "head", (0,))]
                + [
                    (f"q{i:02d}", "maw", (0,), (int(i),))
                    for i in np.linspace(0, jwell.size - 1, 9, dtype=int)
                ]
            },
        )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc",
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

# ### MODFLOW-USG comparison
#
# The same well is represented with the Connected Linear Network Process of
# MODFLOW-USG, once as a single CLN cell, which gives the well one head, and
# once as a network of CLN cells connected end to end, which resolves head loss
# along the borehole. Both use the connection option that computes the leakance
# from the conductivity and thickness of the filter pack, the counterpart of the
# MEAN conductance equation used for the multi-aquifer well. The results are
# stored in a file so that this example runs without MODFLOW-USG.

# +
cln_file = data_path / "mfusg-cln.csv"
cln_discharge_file = data_path / "mfusg-cln-discharge.csv"
cln_map_file = data_path / "mfusg-cln-maps.npz"

# hydraulic conductivity of a conduit is this factor times the radius squared
conduit_k = 9.81 * 86400.0 / (8.0 * 1.787e-6)


def build_mfusg_models(ws, network=False):
    from flopy.mfusg import MfUsg, MfUsgCln, MfUsgDis, MfUsgLpf, MfUsgSms, MfUsgWel

    name = "hwell-network" if network else "hwell-single"
    model = MfUsg(modelname=name, model_ws=str(ws), exe_name="mfusg", structured=True)
    MfUsgDis(
        model,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        nper=nper,
        perlen=perlen,
        nstp=nstp,
        tsmult=tsmult,
        steady=[False],
        itmuni=4,
        lenuni=2,
    )
    flopy.modflow.ModflowBas(model, ibound=1, strt=strt)
    MfUsgLpf(model, laytyp=0, layvka=0, hk=k11, vka=k33, ss=ss, sy=0.0, ipakcb=53)
    MfUsgSms(
        model,
        hclose=hclose,
        hiclose=rclose,
        mxiter=nouter,
        iter1=ninner,
        linmeth=1,
        iacl=2,
        norder=1,
        level=1,
        north=7,
    )
    flopy.modflow.ModflowOc(
        model,
        stress_period_data={(0, i): ["save head", "save budget"] for i in range(nstp)},
        compact=True,
    )

    nnodes = jwell.size if network else 1
    # horizontal conduit, treated as linear so it stays full
    node_properties = [
        [
            i + 1,
            1,
            1,
            delr[jwell[i]] if network else delr[jwell].sum(),
            screen_elevation - well_radius,
            0.0,
            1,
            0,
        ]
        for i in range(nnodes)
    ]
    # IFCON 3 computes the leakance from the conductivity and thickness of the
    # filter pack, as the MEAN conductance equation does in MODFLOW 6
    connections = [
        [
            (i if network else 0) + 1,
            kwell + 1,
            iwell + 1,
            j + 1,
            3,
            k_skin,
            delr[j],
            skin_radius - well_radius,
            0,
        ]
        for i, j in enumerate(jwell)
    ]
    MfUsgCln(
        model,
        ncln=1,
        iclnnds=-1,
        nndcln=[nnodes],
        nclngwc=len(connections),
        node_prop=node_properties,
        cln_gwc=connections,
        cln_circ=[[1, well_radius, conduit_k]],
        ibound=1,
        strt=strt,
        unitnumber=[71, 53, 72, 0, 0, 0, 0],
    )
    # pumped at the center of the screen so the geometry stays symmetric
    if network:
        center = nnodes // 2
        spd = [[center - 1, -0.5 * pumping_rate], [center, -0.5 * pumping_rate]]
    else:
        spd = [[0, -pumping_rate]]
    MfUsgWel(model, ipakcb=0, cln_stress_period_data={0: spd})
    return model


def read_cln_heads(path, nnodes):
    """Return heads from a MODFLOW-USG CLN head file."""
    header = np.dtype(
        [
            ("kstp", "<i4"),
            ("kper", "<i4"),
            ("pertim", "<f4"),
            ("totim", "<f4"),
            ("text", "S16"),
            ("n1", "<i4"),
            ("n2", "<i4"),
            ("ilay", "<i4"),
        ]
    )
    values = []
    with open(path, "rb") as f:
        while True:
            record = f.read(header.itemsize)
            if len(record) < header.itemsize:
                break
            values.append(np.frombuffer(f.read(4 * nnodes), "<f4").copy())
    return np.array(values)


def run_mfusg_models(silent=True):
    """Build, run, and store the MODFLOW-USG results."""
    data_path.mkdir(parents=True, exist_ok=True)
    columns, discharge, maps = {}, None, {}
    for network in (False, True):
        tag = "network" if network else "single"
        ws = workspace / sim_name / f"mfusg-{tag}"
        model = build_mfusg_models(ws, network)
        model.write_input()
        success, buff = model.run_model(silent=silent, report=True)
        assert success, buff

        head = flopy.utils.HeadFile(ws / f"{model.name}.hds")
        kstpkper = head.get_kstpkper()
        columns["totim"] = np.array(head.get_times())
        for n, (x, y, z, _) in enumerate(observations):
            k = int(np.argmin(np.abs(zcenters - z)))
            i = int(np.argmin(np.abs(ycenters - y)))
            j = int(np.argmin(np.abs(xcenters - x)))
            columns[f"{tag}_obs{n}"] = np.array(
                [strt - head.get_data(kstpkper=kk)[k, i, j] for kk in kstpkper]
            )
        steps = [
            int(np.argmin(np.abs(np.array(head.get_times()) - t))) for t in map_times
        ]
        maps[tag] = np.array(
            [
                (strt - head.get_data(kstpkper=kstpkper[step])[kwell])[
                    np.ix_(map_rows, map_cols)
                ]
                for step in steps
            ],
            dtype=np.float32,
        )
        nnodes = jwell.size if network else 1
        cln = read_cln_heads(ws / f"{model.name}.clnhd", nnodes)
        columns[f"{tag}_well"] = strt - cln.mean(axis=1)
        if not network:
            budget = flopy.utils.CellBudgetFile(ws / f"{model.name}.cbc")
            discharge = np.array(
                [r[1] for r in budget.get_data(kstpkper=kstpkper[-1], text="GWF")[0]]
            )

    names = list(columns.keys())
    np.savetxt(
        cln_file,
        np.column_stack([columns[c] for c in names]),
        delimiter=",",
        header=",".join(names),
        comments="",
        fmt="%.6e",
    )
    np.savetxt(
        cln_discharge_file,
        np.column_stack((xcenters[jwell], discharge)),
        delimiter=",",
        header="x,discharge",
        comments="",
        fmt="%.6e",
    )
    np.savez_compressed(cln_map_file, **maps)


def load_mfusg_results():
    """Return the stored MODFLOW-USG drawdown and discharge."""
    with open(cln_file) as f:
        names = f.readline().strip().split(",")
    values = np.loadtxt(cln_file, delimiter=",", skiprows=1)
    cln = {name: values[:, n] for n, name in enumerate(names)}
    discharge = np.loadtxt(cln_discharge_file, delimiter=",", skiprows=1)
    maps = np.load(cln_map_file)
    return cln, discharge[:, 0], discharge[:, 1], {k: maps[k] for k in maps.files}


# -

# ### Plotting results
#
# Define functions to plot model results.

# +
# Figure properties
figure_size = (6.3, 3.4)
map_figure_size = (6.9, 2.65)
colors = ("black", "#1f6fb4", "#b03a2e")


def _head_file(key, suffix=".hds"):
    name = list(parameters.keys())[key]
    return flopy.utils.HeadFile(workspace / sim_name / name / f"{name}{suffix}")


def _drawdown_series(head, k, i, j):
    return np.array(
        [strt - head.get_data(kstpkper=kk)[k, i, j] for kk in head.get_kstpkper()]
    )


def plot_comparison(silent=True):
    wel, maw = _head_file(0), _head_file(1)
    times = np.array(wel.get_times())
    cln, xdischarge, cln_discharge, _ = load_mfusg_results()

    with styles.USGSPlot():
        fig, axes = plt.subplots(
            ncols=2, nrows=1, figsize=figure_size, constrained_layout=True
        )

        ax = axes[0]
        marks = np.array([8, 20, 31, 40, 48, 55, 59])
        for n, (x, y, z, _) in enumerate(observations):
            k = int(np.argmin(np.abs(zcenters - z)))
            i = int(np.argmin(np.abs(ycenters - y)))
            j = int(np.argmin(np.abs(xcenters - x)))
            # evaluate at the cell center the model reports, measured from the
            # well, so the comparison is not biased by the expanding cells
            analytical = np.array(
                [
                    drawdown_analytical(xcenters[j], yrows[i] - ywell, zcenters[k], t)
                    for t in times
                ]
            )
            ax.plot(times, analytical, color=colors[n], lw=1.0, zorder=3)
            for series, marker, size in (
                (_drawdown_series(wel, k, i, j), "o", 4.5),
                (_drawdown_series(maw, k, i, j), "s", 4.5),
                (cln[f"single_obs{n}"], "+", 5.5),
                (cln[f"network_obs{n}"], "x", 4.5),
            ):
                ax.plot(
                    times[marks],
                    series[marks],
                    marker,
                    ms=size,
                    mfc="none",
                    mec=colors[n],
                    mew=0.9,
                    ls="none",
                    zorder=4,
                )
        ax.set_xscale("log")
        ax.set_xlim(1e-3, perlen)
        ax.set_ylim(6.0, 0.0)
        styles.xlabel(ax=ax, label="Time, in days")
        styles.ylabel(ax=ax, label="Drawdown, in meters")
        styles.heading(ax=ax, idx=0)
        styles.remove_edge_ticks(ax=ax)

        handles = [
            Line2D([], [], color=c, lw=1.0, label=f"Zhan and others (2001), {lab}")
            for c, (_, _, _, lab) in zip(colors, observations)
        ]
        handles += [
            Line2D(
                [],
                [],
                ls="none",
                marker=m,
                ms=s,
                mfc="none",
                mec="black",
                mew=0.9,
                label=lab,
            )
            for m, s, lab in (
                ("o", 4.5, "MODFLOW 6, WEL"),
                ("s", 4.5, "MODFLOW 6, MAW"),
                ("+", 5.5, "MODFLOW-USG, one CLN node"),
                ("x", 4.5, "MODFLOW-USG, CLN network"),
            )
        ]
        styles.graph_legend(
            ax=ax,
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc="lower left",
            fontsize=6,
        )

        ax = axes[1]
        widths = delr[jwell]
        maw_name = list(parameters.keys())[1]
        budget = flopy.utils.CellBudgetFile(
            workspace / sim_name / maw_name / f"{maw_name}.cbc"
        )
        maw_discharge = -np.array(
            [
                r[2]
                for r in budget.get_data(
                    kstpkper=budget.get_kstpkper()[-1], text="MAW"
                )[0]
            ]
        )
        ax.plot(
            xcenters[jwell],
            np.full(jwell.size, pumping_rate / screen_length),
            color=colors[0],
            lw=1.0,
            label="Zhan and others (2001), and MODFLOW 6 WEL",
        )
        ax.plot(
            xcenters[jwell],
            maw_discharge / widths,
            "s",
            ms=4.0,
            mfc="none",
            mec=colors[1],
            mew=0.9,
            label="MODFLOW 6, MAW",
        )
        ax.plot(
            xdischarge,
            cln_discharge / widths,
            "+",
            ms=5.0,
            mec=colors[2],
            mew=0.9,
            label="MODFLOW-USG, one CLN node",
        )
        ax.set_xlim(-0.5 * screen_length, 0.5 * screen_length)
        ax.set_ylim(0.0, 30.0)
        styles.xlabel(ax=ax, label="Distance along the screen, in meters")
        styles.ylabel(
            ax=ax,
            label="Discharge to the well, in cubic\nmeters per day per meter of screen",
        )
        styles.heading(ax=ax, idx=1)
        styles.graph_legend(ax=ax, loc="lower center", fontsize=6)
        styles.remove_edge_ticks(ax=ax)

        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig(figs_path / f"{sim_name}-compare.png", dpi=300)


def analytical_field(times, z, xs, ys):
    """Return analytical drawdown maps, using symmetry about the well."""
    ax_, ay = np.unique(np.round(np.abs(xs), 6)), np.unique(np.round(np.abs(ys), 6))
    field = np.empty((len(times), ys.size, xs.size))
    for n, t in enumerate(times):
        lookup = np.array([[drawdown_analytical(x, y, z, t) for x in ax_] for y in ay])
        jx = np.searchsorted(ax_, np.round(np.abs(xs), 6))
        iy = np.searchsorted(ay, np.round(np.abs(ys), 6))
        field[n] = lookup[np.ix_(iy, jx)]
    return field


def plot_difference_maps(silent=True):
    # y is measured from the well, which lies half a cell off the grid center
    xs, ys = xcenters[map_cols], yrows[map_rows] - ywell
    xe = xedges[map_cols[0] : map_cols[-1] + 2]
    ye = yedges[::-1][map_rows[0] : map_rows[-1] + 2] - ywell

    wel, maw = _head_file(0), _head_file(1)
    times = np.array(wel.get_times())
    steps = [int(np.argmin(np.abs(times - t))) for t in map_times]
    analytical = analytical_field([times[s] for s in steps], zcenters[kwell], xs, ys)
    _, _, _, cln_maps = load_mfusg_results()

    def simulated(head, step):
        data = strt - head.get_data(kstpkper=head.get_kstpkper()[step])[kwell]
        return data[np.ix_(map_rows, map_cols)]

    with styles.USGSPlot():
        mosaic = [[f"{c}{r}" for c in "abcde"] for r in range(len(steps))]
        mosaic.append(["cbs"] + ["cbd"] * 4)
        fig, axd = plt.subplot_mosaic(
            mosaic,
            figsize=map_figure_size,
            constrained_layout=True,
            height_ratios=[1.0] * len(steps) + [0.10],
        )
        panel = 0
        for r, step in enumerate(steps):
            reference = analytical[r]
            cases = (
                ("MODFLOW 6,\nWEL", simulated(wel, step)),
                ("MODFLOW 6,\nMAW", simulated(maw, step)),
                ("MODFLOW-USG,\none CLN node", cln_maps["single"][r]),
                ("MODFLOW-USG,\nCLN network", cln_maps["network"][r]),
            )
            ax = axd[f"a{r}"]
            mesh = ax.pcolormesh(
                xe,
                ye,
                reference[::-1],
                cmap="Blues",
                vmin=0.0,
                vmax=8.0,
                rasterized=True,
            )
            contours = ax.contour(
                xs,
                ys[::-1],
                reference[::-1],
                levels=[0.25, 0.5, 1, 2, 4, 6],
                colors="0.25",
                linewidths=0.4,
            )
            ax.clabel(contours, fmt="%g", fontsize=4, inline_spacing=1)
            drawn = [(ax, "Zhan and others\n(2001)")]
            for c, (label, values) in zip("bcde", cases):
                diff = axd[f"{c}{r}"].pcolormesh(
                    xe,
                    ye,
                    (values - reference)[::-1],
                    cmap="RdBu_r",
                    vmin=-0.4,
                    vmax=0.4,
                    rasterized=True,
                )
                drawn.append((axd[f"{c}{r}"], label))
            for ax, label in drawn:
                ax.plot(
                    [-0.5 * screen_length, 0.5 * screen_length],
                    [0, 0],
                    color="0.2",
                    lw=1.2,
                    solid_capstyle="butt",
                )
                ax.set_xlim(-map_extent[0], map_extent[0])
                ax.set_ylim(-map_extent[1], map_extent[1])
                ax.set_aspect("equal")
                ax.tick_params(labelsize=6)
                styles.heading(
                    ax=ax, idx=panel, heading=label if r == 0 else None, fontsize=7
                )
                styles.remove_edge_ticks(ax=ax)
                if r < len(steps) - 1:
                    ax.set_xticklabels([])
                else:
                    styles.xlabel(ax=ax, label="$x$, in meters", fontsize=7)
                panel += 1
            for c in "bcde":
                axd[f"{c}{r}"].set_yticklabels([])
            styles.ylabel(
                ax=axd[f"a{r}"],
                label=f"$t =$ {times[step]:.2f} days\n$y$, in meters",
                fontsize=7,
            )

        bar = fig.colorbar(mesh, cax=axd["cbs"], orientation="horizontal")
        bar.set_label("Drawdown, in meters", fontsize=7)
        bar.ax.tick_params(labelsize=6)
        bar = fig.colorbar(
            diff, cax=axd["cbd"], orientation="horizontal", extend="both"
        )
        bar.set_label(
            "Simulated drawdown minus analytical drawdown, in meters", fontsize=7
        )
        bar.ax.tick_params(labelsize=6)

        if plot_show:
            plt.show()
        if plot_save:
            fig.savefig(figs_path / f"{sim_name}-map.png", dpi=300)


def plot_results(silent=True):
    if not plot:
        return
    plot_comparison(silent=silent)
    plot_difference_maps(silent=silent)


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

# Run the specified-flux representation of the well.

scenario(0)

# Run the multi-aquifer well representation.

scenario(1)

# Rebuild the MODFLOW-USG results, if requested.

if build_usg and run:
    run_mfusg_models()

# Plot the results.

if plot:
    plot_results()
