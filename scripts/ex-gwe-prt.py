# ## Thermal Profile along a Particle Path

# Using a Voronoi grid, this example demonstrates the combined use of GWF, GWE, and PRT.  A steady flow field is established, followed by the establishment of a steady-state temperature field.  Two particles are then routed through the model domain and the temperature of the particle is determined along the flow path.

# ### Initial setup
#
# Import dependencies, define the example name and workspace, and read settings from environment variables.

# +
from pathlib import Path
from pprint import pformat

import flopy
import git
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flopy.discretization import VertexGrid
from flopy.mf6 import MFSimulation
from flopy.utils import GridIntersect
from flopy.utils.triangle import Triangle
from flopy.utils.voronoi import VoronoiGrid
from matplotlib.collections import LineCollection
from modflow_devtools.misc import get_env, timed
from scipy.interpolate import CloughTocher2DInterpolator
from shapely.geometry import LineString, Point

# -

# +
# Example name and workspace paths. If this example is running
# in the git repository, use the folder structure described in
# the README. Otherwise just use the current working directory.
sim_name = "ex-gwe-prt"
gwf_name = sim_name + "-gwf"
gwe_name = sim_name + "-gwe"
prt_name = sim_name + "-prt"
try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None
data_path = root / "data" / sim_name if root else Path.cwd()
workspace = root / "examples" if root else Path.cwd()
figs_path = root / "figures" if root else Path.cwd()
sim_ws = workspace / sim_name
gwf_ws = sim_ws / "gwf"
gwe_ws = sim_ws / "gwe"
prt_ws = sim_ws / "prt"
gwf_ws.mkdir(exist_ok=True, parents=True)
gwe_ws.mkdir(exist_ok=True, parents=True)
prt_ws.mkdir(exist_ok=True, parents=True)

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

# Model units
length_units = "meters"
time_units = "days"

# +
# Model parameters
xmin = 0.0  # Left model domain extent ($m$)
xmax = 2000.0  # Right model domain extent ($m$)
ymin = 0.0  # South model domain extent ($m$)
ymax = 1000.0  # North model domain extent ($m$)
top = 1.0  # Top of the model grid ($m$)
botm = 0.0  # Bottom of the model grid ($m$)
angle_min = 30  # Minimum angle of interior angle of a Voronoi grid cell ($^{\circ}$)
area_max = 1000.0  # Maximum area of a Voronoi grid cell ($m^2$)
nlay = 1  # Number of layers ($-$)
porosity = 0.1  # Porosity ($-$)
strt_temp = 10.0  # Starting temperature of model domain ($^{\circ}C$)
scheme = "TVD"  # Advection scheme ($-$)
alh = 0.0  # Longitudinal mechanical dispersion term ($m$)
ath1 = 0.0  # Transverse mechanical dispersion term ($m$)
ktw = 0.56  # Thermal conductivity of water ($\frac{W}{m \cdot ^{\circ} C}$)
kts = 2.5  # Thermal conductivity of aquifer material ($\frac{W}{m \cdot ^{\circ} C}$)
rhow = 1000  # Density of water ($kg/m^3$)
cpw = 4180.0  # Heat capacity of water ($\frac{J}{kg \cdot ^{\circ} C}$)
rhos = 2650.0  # Density of dry solid aquifer material ($kg/m^3$)
cps = 900.0  # Heat capacity of dry solid aquifer material ($\frac{J}{kg \cdot ^{\circ} C}$)

# Values that do not show up in the table generated for latex
delr = area_max**0.5
ncol = xmax / delr
nrow = ymax / delr
botm = [botm]
nodes = ncol * nrow
lhv = 2500.0
rpts = [[20, i, 0.5] for i in range(1, 999, 20)]

# Convert kts and ktw units to use days
ktw = ktw * 86400
kts = kts * 86400

# -

# ### Solver Parameters
nouter = 1000
ninner = 200
hclose = 1e-6
rclose = 1e-6
relax = 1.0


def get_grid(workspace):
    workspace.mkdir(exist_ok=True, parents=True)
    tri = Triangle(
        maximum_area=area_max,
        angle=angle_min,
        model_ws=workspace,
        exe_name="triangle",
    )
    poly = np.array(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
    tri.add_polygon(poly)
    tri.build(verbose=False)
    return VoronoiGrid(tri)


def buildout_grid(ws):
    grid = get_grid(ws / "grid")
    vgrid = VertexGrid(**grid.get_gridprops_vertexgrid(), nlay=1)
    gi = GridIntersect(vgrid)

    return grid, vgrid, gi


def build_gwf_sim(name):
    global cell2d

    # create grid
    grid, vgrid, gi = buildout_grid(gwf_ws)

    # identify cells on left edge
    line = LineString([(xmin, ymin), (xmin, ymax)])
    cells_left = gi.intersect(line)["cellids"]
    cells_left = np.array(list(cells_left))

    # identify cells on right edge
    line = LineString([(xmax, ymin), (xmax, ymax)])
    cells_right = gi.intersect(line)["cellids"]
    cells_right = np.array(list(cells_right))

    # identify cells on bottom edge
    line = LineString([(xmin, ymin), (xmax, ymin)])
    cells_bottom = gi.intersect(line)["cellids"]
    cells_bottom = np.array(list(cells_bottom))

    # identify well cell
    points = [Point((1200, 500)), Point((700, 200)), Point((1600, 700))]
    well_cells = [vgrid.intersect(p.x, p.y) for p in points]

    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=name, version="mf6", exe_name="mf6", sim_ws=gwf_ws
    )
    flopy.mf6.ModflowTdis(sim, time_units=time_units, perioddata=[[1.0, 1, 1.0]])
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwf_name, save_flows=True)
    flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="complex",
        outer_dvclose=1.0e-8,
        inner_dvclose=1.0e-8,
        filename=f"{gwf_name}.ims",
    )
    flopy.mf6.ModflowGwfdisv(
        gwf, nlay=nlay, **grid.get_disv_gridprops(), top=top, botm=botm
    )

    # k, j, q
    wells = [  # if 5.0 is changed to 0.05 TDIS must be 1e6 250 1.01
        [0, c, -0.05, 80.0] for c in well_cells
    ]
    wells[0][0] *= 100
    wells[2][-2] *= -100
    flopy.mf6.ModflowGwfwel(
        gwf,
        auxiliary="TEMPERATURE",
        maxbound=len(wells),
        save_flows=True,
        pname="WEL",
        stress_period_data={0: wells},
        filename=f"{gwf_name}.wel",
    )

    flopy.mf6.ModflowGwfnpf(
        gwf,
        xt3doptions=[(True)],
        k=10.0,
        save_saturation=True,
        save_specific_discharge=True,
        filename=f"{gwf_name}.npf",
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        ss=0,
        sy=0,
        steady_state={0: True},
        pname="STO",
        filename=f"{gwf_name}.sto",
    )

    flopy.mf6.ModflowGwfic(gwf, strt=1.0, pname="IC", filename=f"{gwf_name}.ic")

    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{gwf_name}.cbc",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        filename=f"{gwf_name}.oc",
    )

    chdlist = []
    icpl_seen = []
    cell2d = grid.get_disv_gridprops()["cell2d"]
    for scl, icpl in enumerate(cells_left):
        y_coord = cell2d[icpl][2]
        chdlist.append([(0, icpl), 2.0, 100.0 * y_coord / ymax])
        icpl_seen.append(icpl)
    for icpl in cells_right:
        chdlist.append([(0, icpl), 1.0, 0.0])
        icpl_seen.append(icpl)
    for icpl in cells_bottom:
        if icpl in icpl_seen:
            continue
        chdlist.append([(0, icpl), 1.8, 80.0])
    flopy.mf6.ModflowGwfchd(
        gwf,
        auxiliary="TEMPERATURE",
        stress_period_data=chdlist,
        pname="CHD",
        filename=f"{gwf_name}.chd",
    )

    return sim


def build_gwe_sim(name):
    grid, dummy, gi = buildout_grid(gwe_ws)

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=gwe_ws, exe_name="mf6")

    flopy.mf6.ModflowTdis(sim, time_units=time_units, perioddata=[[1.0e6, 1000, 1.003]])

    gwe = flopy.mf6.MFModel(
        sim,
        model_type="gwe6",
        modelname=gwe_name,
        model_nam_file=f"{gwe_name}.nam",
    )
    imsgwe = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename=f"{gwe_name}.ims",
    )
    sim.register_ims_package(imsgwe, [gwe.name])

    flopy.mf6.ModflowGwedisv(
        gwe,
        nlay=nlay,
        **grid.get_disv_gridprops(),
        top=top,
        botm=botm,
        pname="DISV",
        filename=f"{gwe_name}.disv",
    )
    flopy.mf6.ModflowGweic(gwe, strt=strt_temp, pname="IC", filename=f"{gwe_name}.ic")
    flopy.mf6.ModflowGweadv(gwe, scheme=scheme, pname="ADV", filename=f"{gwe_name}.adv")
    if ktw != 0:
        flopy.mf6.ModflowGwecnd(
            gwe,
            alh=alh,
            ath1=ath1,
            ktw=ktw,
            kts=kts,
            pname="CND-e",
            filename=f"{gwe_name}.cnd",
        )

    flopy.mf6.ModflowGweest(
        gwe,
        porosity=porosity,
        heat_capacity_water=cpw,
        density_water=rhow,
        latent_heat_vaporization=lhv,
        heat_capacity_solid=cps,
        density_solid=rhos,
        pname="EST-e",
        filename=f"{gwe_name}.est",
    )

    sourcerecarray = [
        ("WEL", "AUX", "TEMPERATURE"),
        ("CHD", "AUX", "TEMPERATURE"),
    ]
    flopy.mf6.ModflowGwessm(
        gwe, sources=sourcerecarray, pname="SSM-e", filename=f"{gwe_name}.ssm"
    )
    flopy.mf6.ModflowGweoc(
        gwe,
        budget_filerecord=f"{gwe_name}.cbc",
        temperature_filerecord=f"{gwe_name}.ucn",
        temperatureprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord={0: [("TEMPERATURE", "ALL"), ("BUDGET", "ALL")]},
        printrecord=[("TEMPERATURE", "LAST"), ("BUDGET", "LAST")],
    )

    pd = [
        ("GWFHEAD", Path(f"../{gwf_ws.name}/{gwf_name}.hds"), None),
        ("GWFBUDGET", Path(f"../{gwf_ws.name}/{gwf_name}.cbc"), None),
    ]
    flopy.mf6.ModflowGwefmi(gwe, packagedata=pd)

    return sim


def build_prt_sim(name):
    # create grid
    grid, vgrid, gi = buildout_grid(prt_ws)

    # create simulation
    sim = flopy.mf6.MFSimulation(
        sim_name=name, version="mf6", exe_name="mf6", sim_ws=prt_ws
    )
    flopy.mf6.ModflowTdis(sim, time_units=time_units, perioddata=[[1.0, 1, 1.0]])
    prt = flopy.mf6.ModflowPrt(sim, modelname=prt_name)
    flopy.mf6.ModflowGwfdisv(
        prt, nlay=nlay, **grid.get_disv_gridprops(), top=top, botm=botm
    )
    flopy.mf6.ModflowPrtmip(prt, pname="mip", porosity=porosity)

    prpdata = [
        # index, (layer, cell index), x, y, z
        (i, (0, vgrid.intersect(p[0], p[1])), p[0], p[1], p[2])
        for i, p in enumerate([rpts[23], rpts[32]])  # first release point crashes
    ]
    prp_track_file = f"{prt_name}.prp.trk"
    prp_track_csv_file = f"{prt_name}.prp.trk.csv"
    flopy.mf6.ModflowPrtprp(
        prt,
        pname="prp1",
        filename=f"{prt_name}_1.prp",
        nreleasepts=len(prpdata),
        packagedata=prpdata,
        perioddata={0: ["FIRST"]},
        track_filerecord=[prp_track_file],
        trackcsv_filerecord=[prp_track_csv_file],
        boundnames=True,
        stop_at_weak_sink=True,  # currently required for this problem
        exit_solve_tolerance=1e-10,
        extend_tracking=True,
    )
    prt_track_file = f"{prt_name}.trk"
    prt_track_csv_file = f"{prt_name}.trk.csv"
    flopy.mf6.ModflowPrtoc(
        prt,
        pname="oc",
        track_filerecord=[prt_track_file],
        trackcsv_filerecord=[prt_track_csv_file],
    )

    pd = [
        ("GWFHEAD", Path(f"../{gwf_ws.name}/{gwf_name}.hds"), None),
        ("GWFBUDGET", Path(f"../{gwf_ws.name}/{gwf_name}.cbc"), None),
    ]

    flopy.mf6.ModflowPrtfmi(prt, packagedata=pd)
    ems = flopy.mf6.ModflowEms(
        sim,
        pname="ems",
        filename=f"{prt_name}.ems",
    )
    sim.register_solution_package(ems, [prt.name])

    return sim


def build_models():
    gwf_sim = build_gwf_sim(sim_name)
    gwe_sim = build_gwe_sim(sim_name)
    prt_sim = build_prt_sim(sim_name)
    return gwf_sim, gwe_sim, prt_sim


def write_models(*sims, silent=False):
    for sim in sims:
        if isinstance(sim, MFSimulation):
            sim.write_simulation(silent=silent)
        else:
            sim.write_input()


@timed
def run_models(*sims, silent=False):
    for sim in sims:
        if isinstance(sim, MFSimulation):
            success, buff = sim.run_simulation(silent=silent, report=True)
        else:
            success, buff = sim.run_model(silent=silent, report=True)
        assert success, pformat(buff)


def temp_interp(pls, temperatures):
    X = []
    Y = []
    Z = []
    # Match up X, Y locations with the state-steady temperature field
    for grdcell in cell2d:
        cellid = grdcell[0]
        X.append(grdcell[1])
        Y.append(grdcell[2])
        Z.append(temperatures[cellid])

    interp = CloughTocher2DInterpolator(list(zip(X, Y)), Z)
    partIDs = pls.name.unique()
    part1 = pls[pls["name"] == partIDs[0]].copy()
    part2 = pls[pls["name"] == partIDs[1]].copy()
    for index, row in part1.iterrows():
        x_prt = row["x"]
        y_prt = row["y"]
        therm = interp(x_prt, y_prt)
        part1.loc[index, "therm"] = therm

    for index, row in part2.iterrows():
        x_prt = row["x"]
        y_prt = row["y"]
        therm = interp(x_prt, y_prt)
        part2.loc[index, "therm"] = therm

    thermDF = pd.concat([part1, part2], ignore_index=True)

    return thermDF


def plot_results(gwf_sim, gwe_sim, prt_sim, silent=True):
    # get gwf output
    gwf = gwf_sim.get_model()
    gwe = gwe_sim.get_model()
    temperatures = gwe.output.temperature().get_data()
    bdobj = gwf.output.budget()
    spdis = bdobj.get_data(text="DATA-SPDIS")[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    # get prt output
    prt_track_csv_file = f"{prt_name}.trk.csv"
    pls = pd.read_csv(prt_ws / prt_track_csv_file, na_filter=False)

    pls = temp_interp(pls, temperatures[0, 0])
    plot_2d = True
    if plot_2d:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(6, 7),
            tight_layout=True,
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )
        ax = axes[0]
        ax.set_xlim([0, 2000])
        ax.set_ylim([0, 1000])

        pmv = flopy.plot.PlotMapView(model=gwf, ax=ax)
        ax.set_aspect("equal")
        pmv.plot_grid(alpha=0.25)
        pmv.plot_ibound(alpha=0.5)
        tempmesh = pmv.plot_array(temperatures, alpha=0.7)
        cv = pmv.contour_array(temperatures, levels=np.linspace(0, 80, 9))
        plt.clabel(cv, colors="k")
        plt.colorbar(
            tempmesh,
            shrink=0.5,
            ax=ax,
            label="Temperature",
            location="bottom",
            fraction=0.1,
        )
        handles = [
            mpl.lines.Line2D(
                [0],
                [0],
                marker=">",
                linestyle="",
                label="Specific discharge",
                color="grey",
                markerfacecolor="gray",
            ),
        ]

        handles.append(
            mpl.lines.Line2D(
                [0], [0], marker="o", linestyle="", label="Well", markerfacecolor="red"
            )
        )
        ax.legend(handles=handles, loc="lower right")
        pmv.plot_vector(qx, qy, normalize=True, alpha=0.25)
        pmv.plot_bc(ftype="WEL")
        mf6_plines = pls.groupby(["iprp", "irpt", "trelease"])
        for ipl, ((iprp, irpt, trelease), pl) in enumerate(mf6_plines):
            pl.plot(
                # title=title,
                kind="line",
                linestyle="--",
                marker="o",
                markersize=0,
                x="x",
                y="y",
                xlabel="X",
                ylabel="Y",
                ax=ax,
                legend=False,
                color="blue",
            )
            if ipl == 0:
                ax.annotate(
                    "Particle 1",
                    xy=(1050, 380),
                    xycoords="data",
                    xytext=(30, -20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="1.0", alpha=0.66),
                    arrowprops=dict(
                        arrowstyle="->",
                        shrinkA=0,
                        shrinkB=5,
                        connectionstyle="angle,angleA=0,angleB=135,rad=40",
                    ),
                )
            else:
                ax.annotate(
                    "Particle 2",
                    xy=(1050, 610),
                    xycoords="data",
                    xytext=(-75, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="1.0", alpha=0.66),
                    arrowprops=dict(
                        arrowstyle="->",
                        shrinkA=0,
                        shrinkB=5,
                        connectionstyle="angle,angleA=0,angleB=135,rad=30",
                    ),
                )

        # Setup 2 additional plots relating the particle's x-position with temperature
        # and alternatively the total travel time with temperature.
        part1 = pls[pls["name"] == pls.name.unique()[0]][["x", "therm"]]
        part2 = pls[pls["name"] == pls.name.unique()[1]][["x", "therm"]]
        part1t = pls[pls["name"] == pls.name.unique()[0]][["t", "therm"]]
        part2t = pls[pls["name"] == pls.name.unique()[1]][["t", "therm"]]

        # Create a set of line segments so that they may be colored individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points1 = np.array([part1["x"], part1["therm"]]).T.reshape(-1, 1, 2)
        points1t = np.array([part1t["t"], part1["therm"]]).T.reshape(-1, 1, 2)
        points2 = np.array([part2["x"], part2["therm"]]).T.reshape(-1, 1, 2)
        points2t = np.array([part2t["t"], part2["therm"]]).T.reshape(-1, 1, 2)
        segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
        segments1t = np.concatenate([points1t[:-1], points1t[1:]], axis=1)
        segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
        segments2t = np.concatenate([points2t[:-1], points2t[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        pad = 5
        norm = plt.Normalize(
            np.min([part1["therm"].min(), part2["therm"].min()]) - pad,
            np.max([part1["therm"].max(), part2["therm"].max()]) + pad,
        )
        lc1 = LineCollection(segments1, norm=norm)
        lc1t = LineCollection(segments1t, norm=norm)
        lc2 = LineCollection(segments2, norm=norm)
        lc2t = LineCollection(segments2t, norm=norm)

        # Set the values used for colormapping
        lc1.set_array(part1["therm"])
        lc1t.set_array(part1t["therm"])
        lc2.set_array(part2["therm"])
        lc2t.set_array(part2t["therm"])
        lc1.set_linewidth(3)
        lc1t.set_linewidth(3)
        lc2.set_linewidth(3)
        lc2t.set_linewidth(3)
        line1 = axes[1].add_collection(lc1)
        line2 = axes[1].add_collection(lc2)
        axes[1].annotate("Particle 2", xy=(400, 68), xycoords="data")
        axes[1].annotate("Particle 1", xy=(400, 50), xycoords="data")
        axes[1].set_xlabel("X ($m$)")
        axes[1].set_xlim(0, 2000)
        xticks = np.arange(0, 2100, 250)
        axes[1].set_xticks(xticks)
        axes[1].set_ylabel(r"Temperature, $^{\circ}C$")
        axes[1].set_ylim(40, 80.0)
        yticks = np.arange(40, 81, 10)
        axes[1].set_yticks(yticks)

        # Third plot - particle travel time vs temperature
        line1t = axes[2].add_collection(lc1t)
        line2t = axes[2].add_collection(lc2t)
        axes[2].annotate("Particle 2", xy=(15000, 68), xycoords="data")
        axes[2].annotate("Particle 1", xy=(15000, 50), xycoords="data")
        axes[2].set_xlabel("Time ($days$)")
        axes[2].set_xlim(0, np.max([part1t["t"].max(), part2t["t"].max()]))
        axes[2].set_ylabel(r"Temperature, $^{\circ}C$")
        axes[2].set_ylim(40.0, 80.0)
        axes[2].set_yticks(yticks)

        # show and/or save figure
        if plot_show:
            plt.show()
        if plot_save:
            fpth = figs_path / f"{sim_name}.png"
            fig.savefig(fpth, dpi=600)


# ### Running the example
#
# Define a function to run the example scenarios and plot results.


# +
def scenario(silent=False):
    gwf_sim, gwe_sim, prt_sim = build_models()
    if write:
        write_models(gwf_sim, gwe_sim, prt_sim, silent=silent)
    if run:
        run_models(gwf_sim, gwe_sim, prt_sim, silent=silent)
    if plot:
        plot_results(gwf_sim, gwe_sim, prt_sim, silent=silent)


# -


# Run the scenario.

scenario(silent=True)
