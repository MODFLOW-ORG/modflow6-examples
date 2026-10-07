# ## Henry Problem with Transport in the Coastal Portion of the Domain
#
# A coastal cross section divided into two groundwater flow models, with solute
# transport solved only in the coastal model. The inland model is coarse and
# carries flow alone, which is how a regional coastal model can be built when
# most of its domain never becomes brackish. One model holding the same cells
# is run alongside for comparison.

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
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from modflow_devtools.misc import get_env, timed

# Example name and workspace paths. If this example is running
# in the git repository, use the folder structure described in
# the README. Otherwise just use the current working directory.
sim_name = "ex-gwt-henry-mm"
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
plot_save = get_env("PLOT_SAVE", True)
# -

# ### Define parameters
#
# Define model units, parameters and other settings.

# +
# Model units
length_units = "meters"
time_units = "days"

# Model parameters
nper = 1  # Number of periods
nstp = 500  # Number of time steps
perlen = 0.5  # Simulation time length ($d$)
system_length = 2.0  # Length of system ($m$)
top = 1.0  # Top of the model ($m$)
delc = 1.0  # Row width ($m$)
split_x = 0.7  # Location of the model boundary ($m$)
inland_ncol = 14  # Number of columns in the inland model
inland_delr = 0.05  # Inland column width ($m$)
coastal_nlay = 40  # Number of layers in the coastal model
coastal_delz = 0.025  # Coastal layer thickness ($m$)
coarse_delr = 0.025  # Coastal column width landward of the mixing zone ($m$)
fine_delr = 0.010  # Coastal column width over the mixing zone ($m$)
refine_x = 0.9  # Landward limit of the refined columns ($m$)
hydraulic_conductivity = 864.0  # Hydraulic conductivity ($m/d$)
porosity = 0.35  # Porosity (unitless)
diffusion_coefficient = 0.57024  # Diffusion coefficient ($m^2/d$)
inflow = 5.7024  # Freshwater inflow rate ($m^3/d$)
seawater_concentration = 35.0  # Seawater concentration (unitless)
drhodc = 0.7  # Density-concentration slope ($kg/m^3$ per unit concentration)

# Model names
inland_name = "inland"
coastal_name = "coastal"
coastal_gwt_name = "coastal-gwt"
single_name = "single"
single_gwt_name = "single-gwt"

# Solver settings
nouter, ninner = 100, 200
hclose, cclose, rclose, relax = 1e-8, 1e-6, 1e-7, 0.97

# Derived grid geometry
coastal_delr = np.concatenate(
    (
        np.full(round((refine_x - split_x) / coarse_delr), coarse_delr),
        np.full(round((system_length - refine_x) / fine_delr), fine_delr),
    )
)
coastal_ncol = coastal_delr.size
coastal_botm = top - coastal_delz * np.arange(1, coastal_nlay + 1)
# -

# ### Model setup
#
# Define functions to build models, write input files, and run the simulation.


# +
def build_divided(sim_ws):
    """Return the two-model simulation, with transport in the coastal model only."""
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")
    flopy.mf6.ModflowTdis(
        sim, nper=nper, perioddata=((perlen, nstp, 1.0),), time_units=time_units
    )

    gwf_inland = flopy.mf6.ModflowGwf(sim, modelname=inland_name, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf_inland,
        length_units=length_units,
        nlay=1,
        nrow=1,
        ncol=inland_ncol,
        delr=inland_delr,
        delc=delc,
        top=top,
        botm=0.0,
    )
    flopy.mf6.ModflowGwfnpf(
        gwf_inland,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity,
    )
    flopy.mf6.ModflowGwfic(gwf_inland, strt=top)
    flopy.mf6.ModflowGwfwel(
        gwf_inland, stress_period_data=[[(0, 0, 0), inflow]], pname="WEL-1"
    )
    flopy.mf6.ModflowGwfoc(
        gwf_inland,
        head_filerecord=f"{inland_name}.hds",
        budget_filerecord=f"{inland_name}.cbc",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    gwf_coastal = flopy.mf6.ModflowGwf(sim, modelname=coastal_name, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf_coastal,
        length_units=length_units,
        nlay=coastal_nlay,
        nrow=1,
        ncol=coastal_ncol,
        delr=coastal_delr,
        delc=delc,
        top=top,
        botm=coastal_botm,
        xorigin=split_x,
    )
    flopy.mf6.ModflowGwfnpf(
        gwf_coastal,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity,
    )
    flopy.mf6.ModflowGwfic(gwf_coastal, strt=top)
    flopy.mf6.ModflowGwfbuy(
        gwf_coastal,
        packagedata=[(0, drhodc, 0.0, coastal_gwt_name, "CONCENTRATION")],
    )
    flopy.mf6.ModflowGwfchd(
        gwf_coastal,
        stress_period_data=[
            [(k, 0, coastal_ncol - 1), top, seawater_concentration]
            for k in range(coastal_nlay)
        ],
        auxiliary="CONCENTRATION",
        pname="CHD-1",
    )
    flopy.mf6.ModflowGwfoc(
        gwf_coastal,
        head_filerecord=f"{coastal_name}.hds",
        budget_filerecord=f"{coastal_name}.cbc",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    gwt = flopy.mf6.ModflowGwt(sim, modelname=coastal_gwt_name, save_flows=True)
    flopy.mf6.ModflowGwtdis(
        gwt,
        length_units=length_units,
        nlay=coastal_nlay,
        nrow=1,
        ncol=coastal_ncol,
        delr=coastal_delr,
        delc=delc,
        top=top,
        botm=coastal_botm,
        xorigin=split_x,
    )
    flopy.mf6.ModflowGwtic(gwt, strt=initial_concentration_array())
    flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")
    flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, diffc=diffusion_coefficient)
    flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
    flopy.mf6.ModflowGwtssm(gwt, sources=[["CHD-1", "AUX", "CONCENTRATION"]])
    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord=f"{coastal_gwt_name}.ucn",
        saverecord=[("CONCENTRATION", "LAST")],
    )

    add_solutions(sim, [inland_name, coastal_name], [coastal_gwt_name])

    # Buoyancy is active in the coastal model only. The inland model is named
    # first because MODFLOW currently requires the buoyant model to be second.
    #
    # Each connection is (cellid1, cellid2, ihc, cl1, cl2, hwva, angldegx, cdist).
    # An ihc of 2 identifies a vertically staggered horizontal connection, so the
    # thickness of the connection is the overlap of the two cells rather than an
    # average of their thicknesses, and hwva is the width of the connection
    # perpendicular to flow rather than its area. An ihc of 1 would average the
    # full thickness of the two cells and overstate the area of flow by 20 times.
    # angldegx and cdist are required because specific discharge is calculated;
    # cdist is the horizontal distance between the cell centers and excludes the
    # vertical offset between the staggered cells.
    cl1, cl2 = 0.5 * inland_delr, 0.5 * coastal_delr[0]
    flopy.mf6.ModflowGwfgwf(
        sim,
        exgtype="GWF6-GWF6",
        nexg=coastal_nlay,
        exgmnamea=inland_name,
        exgmnameb=coastal_name,
        auxiliary=["ANGLDEGX", "CDIST"],
        exchangedata=[
            [(0, 0, inland_ncol - 1), (k, 0, 0), 2, cl1, cl2, delc, 0.0, cl1 + cl2]
            for k in range(coastal_nlay)
        ],
        save_flows=True,
        filename=f"{sim_name}.gwfgwf",
    )
    flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=coastal_name,
        exgmnameb=coastal_gwt_name,
        filename=f"{sim_name}.gwfgwt",
    )
    return sim


def build_single(sim_ws):
    """Return one model holding the same cells as the two-model simulation."""
    nodes = inland_ncol + coastal_nlay * coastal_ncol
    grid = unstructured_grid()

    sim = flopy.mf6.MFSimulation(
        sim_name=f"{sim_name}-single", sim_ws=sim_ws, exe_name="mf6"
    )
    flopy.mf6.ModflowTdis(
        sim, nper=nper, perioddata=((perlen, nstp, 1.0),), time_units=time_units
    )

    gwf = flopy.mf6.ModflowGwf(sim, modelname=single_name, save_flows=True)
    flopy.mf6.ModflowGwfdisu(gwf, length_units=length_units, **grid)
    flopy.mf6.ModflowGwfnpf(
        gwf, save_specific_discharge=True, icelltype=0, k=hydraulic_conductivity
    )
    flopy.mf6.ModflowGwfic(gwf, strt=top)
    flopy.mf6.ModflowGwfbuy(
        gwf,
        packagedata=[(0, drhodc, 0.0, single_gwt_name, "CONCENTRATION")],
    )
    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=[[(0,), inflow, 0.0]],
        auxiliary="CONCENTRATION",
        pname="WEL-1",
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=[
            [(coastal_node(k, coastal_ncol - 1) - 1,), top, seawater_concentration]
            for k in range(coastal_nlay)
        ],
        auxiliary="CONCENTRATION",
        pname="CHD-1",
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{single_name}.hds",
        budget_filerecord=f"{single_name}.cbc",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    gwt = flopy.mf6.ModflowGwt(sim, modelname=single_gwt_name, save_flows=True)
    flopy.mf6.ModflowGwtdisu(gwt, length_units=length_units, **grid)
    strt = np.zeros(nodes, dtype=float)
    for k in range(coastal_nlay):
        strt[coastal_node(k, coastal_ncol - 1) - 1] = seawater_concentration
    flopy.mf6.ModflowGwtic(gwt, strt=strt)
    flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM")
    flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, diffc=diffusion_coefficient)
    flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
    flopy.mf6.ModflowGwtssm(
        gwt,
        sources=[["CHD-1", "AUX", "CONCENTRATION"], ["WEL-1", "AUX", "CONCENTRATION"]],
    )
    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord=f"{single_gwt_name}.ucn",
        saverecord=[("CONCENTRATION", "LAST")],
    )

    add_solutions(sim, [single_name], [single_gwt_name])
    flopy.mf6.ModflowGwfgwt(
        sim,
        exgtype="GWF6-GWT6",
        exgmnamea=single_name,
        exgmnameb=single_gwt_name,
    )
    return sim


def add_solutions(sim, flow_models, transport_models):
    """Add one solution for the flow models and one for the transport models."""
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        relaxation_factor=relax,
        filename=f"{flow_models[0]}.ims",
    )
    sim.register_ims_package(ims, flow_models)
    imsgwt = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=cclose,
        outer_maximum=nouter * 5,
        inner_maximum=ninner,
        inner_dvclose=rclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        relaxation_factor=relax,
        filename=f"{transport_models[0]}.ims",
    )
    sim.register_ims_package(imsgwt, transport_models)


def initial_concentration_array():
    """Return the initial concentration of the coastal transport model.

    The aquifer starts fresh so that flow across the model boundary is seaward at
    every time step, which is the condition under which the exchange carries the
    correct solute flux. An aquifer that is fresh everywhere stays fresh, so the
    seaward column is given the concentration of seawater.
    """
    strt = np.zeros((coastal_nlay, 1, coastal_ncol), dtype=float)
    for k in range(coastal_nlay):
        strt[k, 0, coastal_ncol - 1] = seawater_concentration
    return strt


def coastal_node(k, j):
    """Return the one-based node number of a coastal cell in the unstructured grid."""
    return inland_ncol + k * coastal_ncol + j + 1


def unstructured_grid():
    """Return the DISU arrays for the inland and coastal cells in one grid."""
    nodes = inland_ncol + coastal_nlay * coastal_ncol
    top_a = np.empty(nodes, dtype=float)
    bot_a = np.empty(nodes, dtype=float)
    area = np.empty(nodes, dtype=float)
    for j in range(inland_ncol):
        top_a[j], bot_a[j], area[j] = top, 0.0, inland_delr * delc
    for k in range(coastal_nlay):
        for j in range(coastal_ncol):
            n = coastal_node(k, j) - 1
            top_a[n] = top - k * coastal_delz
            bot_a[n] = coastal_botm[k]
            area[n] = coastal_delr[j] * delc

    # the seaward or lower cell is listed second, so the outward normal of a
    # horizontal connection runs along +x
    conn = []
    for j in range(inland_ncol - 1):
        conn.append((j + 1, j + 2, 1, 0.5 * inland_delr, 0.5 * inland_delr, delc))
    # staggered, as in the exchange of the two-model simulation
    for k in range(coastal_nlay):
        conn.append(
            (
                inland_ncol,
                coastal_node(k, 0),
                2,
                0.5 * inland_delr,
                0.5 * coastal_delr[0],
                delc,
            )
        )
    for k in range(coastal_nlay):
        for j in range(coastal_ncol - 1):
            conn.append(
                (
                    coastal_node(k, j),
                    coastal_node(k, j + 1),
                    1,
                    0.5 * coastal_delr[j],
                    0.5 * coastal_delr[j + 1],
                    delc,
                )
            )
    for k in range(coastal_nlay - 1):
        for j in range(coastal_ncol):
            conn.append(
                (
                    coastal_node(k, j),
                    coastal_node(k + 1, j),
                    0,
                    0.5 * coastal_delz,
                    0.5 * coastal_delz,
                    coastal_delr[j] * delc,
                )
            )

    rows = [[] for _ in range(nodes)]
    for n, m, ihc, cln, clm, hwva in conn:
        rows[n - 1].append((m, ihc, cln, hwva, 0.0))
        rows[m - 1].append((n, ihc, clm, hwva, 180.0 if ihc > 0 else 0.0))

    iac, ja, ihc_a, cl12, hwva_a, angldegx = [], [], [], [], [], []
    for n in range(nodes):
        entries = sorted(rows[n])
        iac.append(len(entries) + 1)
        for arr, val in (
            (ja, n),
            (ihc_a, 0),
            (cl12, 0.0),
            (hwva_a, 0.0),
            (angldegx, 0.0),
        ):
            arr.append(val)
        for m, h, cl, w, ang in entries:
            ja.append(m - 1)
            ihc_a.append(h)
            cl12.append(cl)
            hwva_a.append(w)
            angldegx.append(ang)

    vertices, cell2d = plan_view(nodes)
    return {
        "nodes": nodes,
        "nja": len(ja),
        "nvert": len(vertices),
        "top": top_a,
        "bot": bot_a,
        "area": area,
        "iac": iac,
        "ja": ja,
        "ihc": ihc_a,
        "cl12": cl12,
        "hwva": hwva_a,
        "angldegx": angldegx,
        "vertices": vertices,
        "cell2d": cell2d,
    }


def plan_view(nodes):
    """Return the vertex and cell2d lists of the unstructured grid.

    The section is one cell wide, so every cell is the same rectangle in plan
    view. The data are needed only so that specific discharge can be computed.
    """
    inland_edges = np.linspace(0.0, inland_ncol * inland_delr, inland_ncol + 1)
    coastal_edges = split_x + np.concatenate(([0.0], np.cumsum(coastal_delr)))
    xedges = np.unique(np.round(np.concatenate((inland_edges, coastal_edges)), 9))
    vertices = []
    for i, x in enumerate(xedges):
        vertices.append([2 * i, float(x), delc])
        vertices.append([2 * i + 1, float(x), 0.0])

    def polygon(x0, x1):
        """Return the four vertex numbers of a cell, listed clockwise."""
        i0 = int(np.searchsorted(xedges, round(x0, 9)))
        i1 = int(np.searchsorted(xedges, round(x1, 9)))
        return [2 * i0, 2 * i1, 2 * i1 + 1, 2 * i0 + 1]

    cell2d = []
    for j in range(inland_ncol):
        x0, x1 = inland_edges[j], inland_edges[j + 1]
        cell2d.append([j, 0.5 * (x0 + x1), 0.5 * delc, 4] + polygon(x0, x1))
    for k in range(coastal_nlay):
        for j in range(coastal_ncol):
            x0, x1 = coastal_edges[j], coastal_edges[j + 1]
            cell2d.append(
                [coastal_node(k, j) - 1, 0.5 * (x0 + x1), 0.5 * delc, 4]
                + polygon(x0, x1)
            )
    return vertices, cell2d


def build_models():
    """Return the two-model and one-model simulations."""
    print(f"Building models...{sim_name}")
    return (
        build_divided(workspace / sim_name),
        build_single(workspace / f"{sim_name}-single"),
    )


def write_models(sims, silent=True):
    for sim in sims:
        sim.write_simulation(silent=silent)


@timed
def run_models(sims, silent=True):
    for sim in sims:
        success, buff = sim.run_simulation(silent=silent, report=True)
        assert success, buff


# -

# ### Plotting results
#
# Define functions to plot model results.

# +
# Figure properties
figure_size = (6.3, 3.1)
grid_figure_size = (6.3, 2.6)
detail_halfwidth = 0.075  # half-width of the portion drawn in the exchange detail
detail_ratio = 0.8  # width of the exchange detail panel, relative to the section
levels = [0.1, 0.5, 0.9]
fresh_center = 0.5 * split_x


def salinity_cmap():
    """Return a white to red color map that keeps black arrows legible."""
    return LinearSegmentedColormap.from_list(
        "salinity", plt.get_cmap("Reds")(np.linspace(0.04, 0.84, 256))
    )


def section_edges():
    """Return the cell edges of the inland and coastal grids."""
    inland = (
        np.linspace(0.0, inland_ncol * inland_delr, inland_ncol + 1),
        np.array([top, 0.0]),
    )
    coastal = (
        split_x + np.concatenate(([0.0], np.cumsum(coastal_delr))),
        np.concatenate(([top], coastal_botm)),
    )
    return inland, coastal


def setup_axes(ax):
    """Apply the shared cross-section limits and labels."""
    ax.set_xlim(0.0, system_length)
    ax.set_ylim(0.0, top)
    ax.set_aspect("equal")
    styles.xlabel(ax=ax, label="Distance, in meters")
    styles.ylabel(ax=ax, label="Elevation, in meters")
    styles.remove_edge_ticks(ax)


def draw_boundary(ax):
    """Draw the outline of each model and the boundary between them."""
    for x0, x1 in ((0.0, split_x), (split_x, system_length)):
        ax.plot(
            [x0, x1, x1, x0, x0],
            [0.0, 0.0, top, top, 0.0],
            color="black",
            lw=0.8,
            zorder=6,
        )
    ax.plot(
        [split_x, split_x],
        [0.0, top],
        color="black",
        lw=1.2,
        ls=(0, (6, 4)),
        zorder=7,
    )


def concentration(sim, name, shape=None):
    """Return the simulated concentration at the end of the simulation."""
    conc = sim.get_model(name).output.concentration()
    data = conc.get_data(totim=conc.get_times()[-1])
    return data.reshape(shape) if shape else data[:, 0, :]


def plot_grid(sims):
    """Plot the two grids and the connections between them."""
    (ixe, ize), (cxe, cze) = section_edges()
    with styles.USGSMap():
        fig = plt.figure(figsize=grid_figure_size, layout="constrained")
        fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.94))
        axs = fig.subplots(
            1, 2, width_ratios=[system_length, detail_ratio], sharey=True
        )

        ax = axs[0]
        ax.set_anchor("N")
        for xe, ze, color in ((ixe, ize, "0.15"), (cxe, cze, "0.55")):
            ax.vlines(xe, ze[-1], ze[0], color=color, lw=0.15, zorder=2)
            ax.hlines(ze, xe[0], xe[-1], color=color, lw=0.15, zorder=2)
        setup_axes(ax)
        draw_boundary(ax)
        ax.add_patch(
            Rectangle(
                (split_x - detail_halfwidth, 0.0),
                2.0 * detail_halfwidth,
                top,
                fc="#1f5fb4",
                ec="#1f5fb4",
                alpha=0.18,
                lw=0.8,
                zorder=8,
            )
        )
        styles.heading(ax, letter="A", heading="Model grids")

        ax = axs[1]
        x1, x2 = split_x - 0.5 * inland_delr, split_x + 0.5 * coastal_delr[0]
        ax.axvspan(split_x - inland_delr, split_x, color="0.88", zorder=0)
        ax.axvspan(split_x, split_x + coastal_delr[0], color="#fbd6cc", zorder=0)
        for xe, ze, color, lw in ((ixe, ize, "0.15", 0.5), (cxe, cze, "0.55", 0.2)):
            ax.vlines(xe, ze[-1], ze[0], color=color, lw=lw, zorder=2)
            ax.hlines(ze, xe[0], xe[-1], color=color, lw=lw, zorder=2)
        for k in range(coastal_nlay):
            ax.plot(
                [x1, x2],
                [0.5 * top, coastal_botm[k] + 0.5 * coastal_delz],
                color="firebrick",
                lw=0.25,
                alpha=0.6,
                zorder=3,
            )
        ax.plot(
            [split_x, split_x],
            [0.0, top],
            color="black",
            lw=1.0,
            ls=(0, (6, 4)),
            zorder=4,
        )
        ax.plot(x1, 0.5 * top, "o", ms=3, color="firebrick", zorder=5)
        ax.set_xlim(split_x - detail_halfwidth, split_x + detail_halfwidth)
        ax.set_xticks([split_x - 0.05, split_x, split_x + 0.05])
        # match the height of panel A, which equal aspect otherwise shortens
        ax.set_box_aspect(top / detail_ratio)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1f5fb4")
        styles.xlabel(ax=ax, label="Distance, in meters")
        styles.remove_edge_ticks(ax)
        styles.heading(ax, letter="B", heading="Exchange detail")

        if plot_save:
            fig.savefig(figs_path / f"{sim_name}-grid.png")
        plt.close(fig)


def plot_conc(sims):
    """Plot the simulated salinity and specific discharge of the two models."""
    sim = sims[0]
    (ixe, ize), (cxe, cze) = section_edges()
    conc = concentration(sim, coastal_gwt_name) / seawater_concentration
    cmap = salinity_cmap()
    norm = BoundaryNorm(np.linspace(0.0, 1.0, 11), cmap.N)

    with styles.USGSMap():
        fig = plt.figure(figsize=figure_size, layout="constrained")
        fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.94))
        ax = fig.add_subplot()
        ax.pcolormesh(ixe, ize, np.zeros((1, inland_ncol)), cmap=cmap, norm=norm)
        mappable = ax.pcolormesh(cxe, cze, conc, cmap=cmap, norm=norm)
        xc = 0.5 * (cxe[:-1] + cxe[1:])
        zc = 0.5 * (cze[:-1] + cze[1:])
        ax.contour(
            xc, zc, conc, levels=levels, colors="black", linewidths=0.6, zorder=4
        )

        qref = draw_vectors(ax, sim)
        ax.quiverkey(
            qref[0],
            fresh_center / system_length,
            0.24,
            qref[1],
            f"{qref[1]:.0f} meters per day",
            labelpos="N",
            labelsep=0.04,
            coordinates="axes",
            fontproperties={"size": 7},
        )
        setup_axes(ax)
        draw_boundary(ax)
        label_models(ax)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.96, pad=0.015, fraction=0.035)
        cbar.set_label("Relative salinity, dimensionless")
        styles.remove_edge_ticks(cbar.ax)
        styles.heading(ax, heading="Simulated salinity and specific discharge")

        if plot_save:
            fig.savefig(figs_path / f"{sim_name}-conc.png")
        plt.close(fig)


def draw_vectors(ax, sim, kstep=3, hstep=7):
    """Draw specific discharge vectors and return the quiver and its reference."""
    (ixe, ize), (cxe, cze) = section_edges()
    fields = []
    for name, xe, ze, hs in (
        (inland_name, ixe, ize, 3),
        (coastal_name, cxe, cze, hstep),
    ):
        model = sim.get_model(name)
        cbc = model.output.budget()
        spdis = cbc.get_data(text="DATA-SPDIS", totim=cbc.get_times()[-1])[0]
        qx, _, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, model)
        fields.append((xe, ze, qx[:, 0, :], qz[:, 0, :], hs))

    qref = float(np.percentile(np.sqrt(fields[1][2] ** 2 + fields[1][3] ** 2), 90.0))
    scale = qref / 0.055
    for xe, ze, qx, qz, hs in fields:
        xc = 0.5 * (xe[:-1] + xe[1:])
        zc = 0.5 * (ze[:-1] + ze[1:])
        xg, zg = np.meshgrid(xc[::hs], zc[::kstep])
        quiv = ax.quiver(
            xg,
            zg,
            qx[::kstep, ::hs],
            qz[::kstep, ::hs],
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.0018,
            headwidth=4.0,
            headlength=4.5,
            headaxislength=4.0,
            color="black",
            zorder=5,
        )
    return quiv, qref


def label_models(ax):
    """Label each model domain inside the cross section."""
    bbox = {
        "boxstyle": "round,pad=0.2",
        "fc": "white",
        "ec": "0.45",
        "lw": 0.5,
        "alpha": 0.93,
    }
    for x, label in (
        (fresh_center, "INLAND MODEL\nflow only"),
        (0.5 * (split_x + system_length), "COASTAL MODEL\nflow and transport"),
    ):
        styles.add_text(
            ax=ax,
            text=label,
            x=x,
            y=0.90,
            transform=False,
            bold=False,
            italic=False,
            fontsize=7,
            ha="center",
            va="center",
            bbox=bbox,
            zorder=9,
        )


def plot_comparison(sims):
    """Compare the salinity contours of the two-model and one-model simulations."""
    (ixe, ize), (cxe, cze) = section_edges()
    divided = concentration(sims[0], coastal_gwt_name) / seawater_concentration
    flat = concentration(sims[1], single_gwt_name, shape=(-1,))
    single = flat[inland_ncol:].reshape(coastal_nlay, coastal_ncol)
    single = single / seawater_concentration
    cmap = salinity_cmap()
    norm = BoundaryNorm(np.linspace(0.0, 1.0, 11), cmap.N)
    xc = 0.5 * (cxe[:-1] + cxe[1:])
    zc = 0.5 * (cze[:-1] + cze[1:])

    with styles.USGSMap():
        fig = plt.figure(figsize=figure_size, layout="constrained")
        fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.94))
        ax = fig.add_subplot()
        ax.pcolormesh(ixe, ize, np.zeros((1, inland_ncol)), cmap=cmap, norm=norm)
        mappable = ax.pcolormesh(cxe, cze, divided, cmap=cmap, norm=norm)
        cs = ax.contour(
            xc, zc, divided, levels=levels, colors="black", linewidths=0.9, zorder=5
        )
        ax.clabel(cs, fmt="%3.1f", fontsize=6, inline_spacing=4)
        ax.contour(
            xc,
            zc,
            single,
            levels=levels,
            colors="#1f5fb4",
            linewidths=0.9,
            linestyles="--",
            zorder=6,
        )
        setup_axes(ax)
        draw_boundary(ax)

        handles = [
            Line2D([0], [0], color="black", lw=0.9, label="two models"),
            Line2D([0], [0], color="#1f5fb4", lw=0.9, ls="--", label="one model"),
        ]
        styles.graph_legend(
            ax,
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc="upper center",
            bbox_to_anchor=(fresh_center / system_length, 0.98),
            fontsize=7,
            frameon=True,
            facecolor="white",
            edgecolor="0.45",
            framealpha=0.93,
        )
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.96, pad=0.015, fraction=0.035)
        cbar.set_label("Relative salinity, dimensionless")
        styles.remove_edge_ticks(cbar.ax)
        styles.heading(ax, heading="Effect of dividing the section")

        if plot_save:
            fig.savefig(figs_path / f"{sim_name}-compare.png")
        plt.close(fig)


def plot_results(sims):
    plot_grid(sims)
    plot_conc(sims)
    plot_comparison(sims)


# -

# ### Running the example
#
# Define and invoke a function to run the example scenario, then plot results.


# +
def scenario(silent=True):
    sims = build_models()
    if write:
        write_models(sims, silent=silent)
    if run:
        run_models(sims, silent=silent)
    if plot:
        plot_results(sims)


scenario()
# -
