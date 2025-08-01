# %% [markdown]
# # Different advection schemes
#
# In this example we are going to look at the available schemes availalbe for the advection package.
# There are 4 schemes available:
# - Upstream
# - Central difference (CD)
# - TVD
# - UTVD
#
# To demonstrate the difference between the different schemes we are going to look at 3 different functions:
# - sin^2 wave
# - block wave
# - step wave
#
# The sin^2 wave is a continious smooth function. When using the CD, TVD and UTVD schemes we are able to achieve 2^nd order accuracy.
# The block wave and step wave are functions that have a discontinuity. This will cause the CD scheme to osscilate. In the worst case it might even get unstable and the simulation might fail. This is not the cause for the Upstream, TVD and UTVD schemes which are all bounded.
#
# Further we are also going to explore the effect of the grid. We are going to look at 3 types of grids:
# - structured
# - triangle
# - voronoi
# We will see that second order accuracy is obtained on the strutured and voronoi grid with the UTVD scheme. On the triangle we get 1st order accuracy. This is becuase the vectors connecting cell centers don't cross the cell boundaries at a right angle. This causes smearing too happen.

# %% [markdown]
# # Initial setup
#
# Import dependencies, define the example name and workspace, and read settings from environment variables.

# %%
from pathlib import Path
import math
import numpy as np
from shapely.geometry import Polygon, LineString
import pandas as pd
import matplotlib.pyplot as plt
import collections.abc
from scipy.interpolate import LinearNDInterpolator
import geopandas as gpd
import itertools

import flopy
from flopy.utils.cvfdutil import get_disv_gridprops
from modflow_devtools.misc import get_env, timed
from flopy.plot.styles import styles
from flopy.utils import GridIntersect
from flopy.utils.triangle import Triangle
from flopy.discretization.vertexgrid import VertexGrid
from flopy.utils.voronoi import VoronoiGrid


# Example name and workspace paths. If this example is running
# in the git repository, use the folder structure described in
# the README. Otherwise just use the current working directory.
sim_name = "ex-gwt-adv-schemes"
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

# %% [markdown]
# # Define parameters
#
# Define model units, parameters and other settings.

# %%
# Cases
grids = ["structured", "triangle", "voronoi"]
schemes = ["upstream", "central", "tvd", "utvd"]
wave_functions = ["sin2-wave", "step-wave", "block-wave"]

# Model units
length_units = "centimeters"
time_units = "seconds"

# Solver settings
nouter = 100
ninner = 300
hclose = 1e-6
cclose = 1e-6
rclose = 1e-6

# Grid parameters
nper = 1  # Number of periods
nlay = 1  # Number of layers
ncol = 50  # Number of columns
nrow = 50  # Number of rows

Length = Width = 100.0 # Length of system ($cm$)

delr = Length / ncol
delc = Width / nrow
top = 1.0  # Top of the model ($cm$)
botm = 0  # Layer bottom elevation ($cm$)

# Model parameter
specific_discharge = 0.5  # Specific discharge ($cm s^{-1}$)
hydraulic_conductivity = 0.01  # Hydraulic conductivity ($cm s^{-1}$)

angle =  math.radians(45) # angle of the flow from the x-axis
qx = math.cos(angle) * specific_discharge # x-component specific discharge
qy = math.sin(angle) * specific_discharge # y-component specific discharge
inlet_height = 20.0

total_time = 50.0  # Simulation time ($s$)
dt = 0.01


# %% [markdown]
# # Analytical solution
#
# Exact solution of the concentration field. We take a uniform flow field along the x-axis and rotated with the flow angle to get the exact solution.

# %%
def exact_solution_concentration(x, y, analytical):
    # Rotate to 1d solution space
    reverse_angle = -angle
    rotated_x = math.cos(reverse_angle) * x - math.sin(reverse_angle) * y
    rotated_y = math.sin(reverse_angle) * x + math.cos(reverse_angle) * y
    
    # Compute concentration
    return inlet_signal(rotated_y, analytical)
        
    
def inlet_signal(y, signal_name):
    clipped_y = np.clip(y, -inlet_height / 2, inlet_height / 2)
    match signal_name:
        case "step-wave":
            return np.where(y < 0, np.ones(y.shape), np.zeros(y.shape))
        case "block-wave":
            return np.where(np.abs(y) < inlet_height/2, np.ones(y.shape), np.zeros(y.shape))
        case "sin2-wave":
            return np.power(np.cos(math.pi * clipped_y /inlet_height),2)
        case _:
            raise ValueError("Unknow signal name")


# %% [markdown]
# # Grid Helper methods

# %%
def grid_triangulator(itri, delr, delc):
    nrow, ncol = itri.shape
    if np.isscalar(delr):
        delr = delr * np.ones(ncol)
    if np.isscalar(delc):
        delc = delc * np.ones(nrow)
    regular_grid = flopy.discretization.StructuredGrid(delc, delr)
    vertdict = {}
    icell = 0
    for i in range(nrow):
        for j in range(ncol):
            vs = regular_grid.get_cell_vertices(i, j)
            if itri[i, j] == 0:
                vertdict[icell] = [vs[0], vs[1], vs[2], vs[3], vs[0]]
                icell += 1
            elif itri[i, j] == 1:
                vertdict[icell] = [vs[0], vs[1], vs[3], vs[0]]
                icell += 1
                vertdict[icell] = [vs[3], vs[1], vs[2], vs[3]]
                icell += 1
            elif itri[i, j] == 2:
                vertdict[icell] = [vs[0], vs[2], vs[3], vs[0]]
                icell += 1
                vertdict[icell] = [vs[0], vs[1], vs[2], vs[0]]
                icell += 1
            else:
                raise Exception(f"Unknown itri value: {itri[i, j]}")
    verts, iverts = flopy.utils.cvfdutil.to_cvfd(vertdict)
    return verts, iverts


def cvfd_to_cell2d(verts, iverts):
    vertices = []
    for i in range(verts.shape[0]):
        x = verts[i, 0]
        y = verts[i, 1]
        vertices.append([i, x, y])
    cell2d = []
    for icell2d, vs in enumerate(iverts):
        points = [tuple(verts[ip]) for ip in vs]
        xc, yc = flopy.utils.cvfdutil.centroid_of_polygon(points)
        cell2d.append([icell2d, xc, yc, len(vs), *vs])
    return vertices, cell2d

def grid_intersector(vgrid):
    # if not isinstance(grid, VoronoiGrid):
    return flopy.utils.GridIntersect(vgrid)

    
def get_boundary(gi, line):        
    line = LineString(line)
    cells = gi.intersect(line)["cellids"]
    cells = np.array(list(cells))
    
    return cells

def create_bc(boundary_ids, value, auxvalue = None):
    if auxvalue is None:
        if isinstance(value, (collections.abc.Sequence, np.ndarray)):
            return [[(0, cell_id), value[idx]] for idx, cell_id in enumerate(boundary_ids)]
        else:
            return [[(0, cell_id), value] for idx, cell_id in enumerate(boundary_ids)]
    else:
        if isinstance(auxvalue, (collections.abc.Sequence, np.ndarray)):
            return [[(0, cell_id), value, auxvalue[idx]] for idx, cell_id in enumerate(boundary_ids)]
        else:
            return [[(0, cell_id), value, auxvalue] for idx, cell_id in enumerate(boundary_ids)]

def merge_bc_sum(boundaries):
    df = pd.DataFrame(boundaries)
    summed = df.groupby([0], as_index=False)[1].sum()
    
    return summed.values.tolist()

def merge_bc_mean(boundaries):
    df = pd.DataFrame(boundaries)
    mean = df.groupby([0], as_index=False)[1].mean()
    
    return mean.values.tolist()


def create_grid(grid_type):
    if grid_type == "structured":
        itri = np.zeros((nrow, ncol), dtype=int)
        verts, iverts = grid_triangulator(itri, delr, delc)
        vertices, cell2d = cvfd_to_cell2d(verts, iverts)

        ncpl = len(cell2d)
        nvert = len(verts)

        return ncpl, nvert, vertices, cell2d
    elif grid_type == "triangle":
        active_domain = [(0, 0), (Length, 0), (Length, Width), (0, Width)]
        tri = Triangle(angle=30, maximum_area=delc*delr * 1.5, model_ws=workspace)
        tri.add_polygon(active_domain)
        tri.build()

        cell2d = tri.get_cell2d()
        vertices = tri.get_vertices()
        ncpl = tri.ncpl
        nvert = tri.nvert

        return ncpl, nvert, vertices, cell2d
    elif grid_type == "voronoi":
        active_domain = [(0, 0), (Length, 0), (Length, Width), (0, Width)]
        tri = Triangle(angle=30, maximum_area=delc * delr / 1.5 * 1.2, model_ws=workspace)
        tri.add_polygon(active_domain)
        tri.build()

        vor = VoronoiGrid(tri)
        disv_gridprops = vor.get_gridprops_vertexgrid()

        cell2d = disv_gridprops["cell2d"]
        vertices = disv_gridprops["vertices"]
        ncpl = disv_gridprops["ncpl"]
        nvert = len(vertices)

        return ncpl, nvert, vertices, cell2d

    else:
        raise f"grid of type '{grid_type}' is not supported."

def axis_aligned_segment_length(polygon, axis='y', value=0):
    total_length = 0.0
    coords = list(polygon.exterior.coords)

    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        if axis == 'y' and p1[1] == value and p2[1] == value:
            segment = LineString([p1, p2])
            total_length += segment.length
        elif axis == 'x' and p1[0] == value and p2[0] == value:
            segment = LineString([p1, p2])
            total_length += segment.length

    return total_length



# %% [markdown]
# # Model setup
#
# Define functions to build models, write input files, and run the simulation.

# %%
def build_mf6gwf(grid_type):
    gwfname = f"flow_{grid_type}"
    sim_ws = workspace / sim_name / Path(gwfname)
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")

    tdis_ds = ((total_time, 1, 1.0),)
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)

    flopy.mf6.ModflowIms(
        sim,
        print_option="ALL",
        linear_acceleration="bicgstab",
        outer_maximum=nouter,
        outer_dvclose=hclose,
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose
        )
    
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True)

    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_specific_discharge=True,
        save_saturation=True,
        export_array_ascii=True,
        icelltype=0,
        k=hydraulic_conductivity,
        # xt3doptions=[(True)]
    )

    flopy.mf6.ModflowGwfic(gwf, strt=0.0, export_array_ascii=True)
   
    ncpl, nvert, vertices, cell2d = create_grid(grid_type)
    disv = flopy.mf6.ModflowGwfdisv(
        gwf,
        nlay=nlay,
        ncpl=ncpl,
        nvert=nvert,
        top=top,
        botm=botm,
        vertices=vertices,
        cell2d=cell2d,
        filename=f"{gwfname}.disv",
    )

    head_filerecord = f"{gwfname}.hds"
    budget_filerecord_csv = f"{gwfname}.bud.csv"
    budget_filerecord = f"{gwfname}.bud"
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=head_filerecord,
        budgetcsv_filerecord=budget_filerecord_csv,
        budget_filerecord=budget_filerecord,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    
    # Boundary lines
    bottom_edge = [(0, 0), (Length, 0)]
    right_edge = [(Length, 0), (Length, Width)]
    top_edge = [(Length, Width), (0, Width)]
    left_edge = [(0, Width), (0, 0)]

    # Identify boundary ids
    vgrid = VertexGrid(vertices=vertices, cell2d=cell2d, ncpl=ncpl, nlay=1)
    gi = grid_intersector(vgrid)
    
    bottom_boundary_ids = get_boundary(gi, bottom_edge)
    right_boundary_ids = get_boundary(gi, right_edge)
    top_boundary_ids = get_boundary(gi, top_edge)
    left_boundary_ids = get_boundary(gi, left_edge)

    # Set top right element to a chd 0. This makes the solution unique
    topright_id, right_boundary_id, top_boundary_id = np.intersect1d(right_boundary_ids, top_boundary_ids, return_indices=True)

    # top_boundary_ids = np.delete(top_boundary_ids, top_boundary_id)
    # right_boundary_ids = np.delete(right_boundary_ids, right_boundary_id)
    
    if topright_id.any():
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data = create_bc(topright_id, 0))
    
    geometry = vgrid.geo_dataframe.geometry
    cell_area = geometry.area.values
    Inflow_Area_left = geometry.loc[left_boundary_ids].apply(lambda poly: axis_aligned_segment_length(poly, axis='x', value=0)).values
    Inflow_Area_bot = geometry.loc[bottom_boundary_ids].apply(lambda poly: axis_aligned_segment_length(poly, axis='y', value=0)).values
    Outflow_Area_right = geometry.loc[right_boundary_ids].apply(lambda poly: axis_aligned_segment_length(poly, axis='x', value=Length)).values
    Outflow_Area_top = geometry.loc[top_boundary_ids].apply(lambda poly: axis_aligned_segment_length(poly, axis='y', value=Width)).values

    flopy.mf6.ModflowGwfrch(gwf,
        stress_period_data = merge_bc_sum(
                                create_bc(left_boundary_ids     , qx * Inflow_Area_left / cell_area[left_boundary_ids]) +
                                create_bc(bottom_boundary_ids   , qy * Inflow_Area_bot / cell_area[bottom_boundary_ids]) +
                                create_bc(right_boundary_ids    ,-qx * Outflow_Area_right / cell_area[right_boundary_ids]) +
                                create_bc(top_boundary_ids      ,-qy * Outflow_Area_top / cell_area[top_boundary_ids])
        ),
    )

    return sim

def build_mf6gwt(grid_type, scheme, wave_func):
    pathname = f"trans_{grid_type}_{wave_func}_{scheme}"
    gwtname = f"trans"
    gwfname = f"flow_{grid_type}"
    sim_ws = workspace / sim_name / Path(pathname)
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")

    nsteps = int(total_time/dt)
    tdis_ds = ((total_time, nsteps, 1.0),)

    tdis = flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)

    flopy.mf6.ModflowIms(
        sim, 
        linear_acceleration="bicgstab", 
        print_option="SUMMARY",
        outer_maximum=nouter,
        outer_dvclose=cclose,
        inner_maximum=ninner,
        inner_dvclose=cclose,
        rcloserecord=rclose,
    )

    gwt = flopy.mf6.ModflowGwt(sim, modelname=gwtname, save_flows=True)

    flopy.mf6.ModflowGwtmst(gwt, porosity=1.0)

    flopy.mf6.ModflowGwtssm(gwt)

    flopy.mf6.ModflowGwtadv(gwt, scheme=scheme)

    packagedata = [
        ("GWFHEAD", f"../{gwfname}/{gwfname}.hds", None),
        ("GWFBUDGET", f"../{gwfname}/{gwfname}.bud", None),
    ]
    flopy.mf6.ModflowGwtfmi(
        gwt, 
        packagedata=packagedata,
        save_flows=True
    )

    flopy.mf6.ModflowGwtoc(
        gwt,
        budget_filerecord=f"{gwtname}.cbc",
        concentration_filerecord=f"{gwtname}.ucn",
        saverecord=[("CONCENTRATION", "FREQUENCY", 10), ("BUDGET", "FREQUENCY", 10)],
        printrecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    )

    ncpl, nvert, vertices, cell2d = create_grid(grid_type)
    disv = flopy.mf6.ModflowGwtdisv(
        gwt,
        nlay=nlay,
        ncpl=ncpl,
        nvert=nvert,
        top=top,
        botm=botm,
        vertices=vertices,
        cell2d=cell2d,
        filename=f"{gwtname}.disv",
    )

    # Boundary lines
    bottom_edge = [(0, 0), (Length, 0)]
    left_edge = [(0, Width), (0, 0)]

    # Identify boundary ids
    vgrid = VertexGrid(vertices=vertices, cell2d=cell2d, ncpl=ncpl, nlay=1)
    gi = grid_intersector(vgrid)
    
    bottom_boundary_ids = get_boundary(gi, bottom_edge)
    left_boundary_ids = get_boundary(gi, left_edge)

    # Compute exact solution for the inlet boundaries
    cell2d_df = pd.DataFrame(cell2d)
    xc, yc = cell2d_df.loc[bottom_boundary_ids,1:2].T.to_numpy()
    conc_bottom = exact_solution_concentration(xc, yc, wave_func)
    xc, yc = cell2d_df.loc[left_boundary_ids,1:2].T.to_numpy()
    conc_left = exact_solution_concentration(xc, yc, wave_func)

    # Set inlet concentrations
    cnc = flopy.mf6.ModflowGwtcnc(
        gwt,
        stress_period_data=merge_bc_mean(
            create_bc(bottom_boundary_ids, conc_bottom) +
            create_bc(left_boundary_ids, conc_left)
        )
    )

    # Set initial condition equal to analytical
    xc, yc = cell2d_df.loc[:,1:2].T.to_numpy()
    conc = exact_solution_concentration(xc, yc, wave_func)
    flopy.mf6.ModflowGwtic(gwt, strt=conc)

    return sim



# %%
def write_models(sim, silent=True):
    sim.write_simulation(silent=silent)


@timed
def run_models(sim, silent=True):
    success, buff = sim.run_simulation(silent=silent)
    assert success, buff


# %% [markdown]
# # Plotting results
#
# Define functions to plot model results.

# %%
def plot_results(gwf_sims, gwt_sims):
    plot_flows(gwf_sims)
    plot_concentrations(gwt_sims)
    plot_concentration_cross_sections(gwt_sims)

def plot_flows(gwf_sims):
    with styles.USGSPlot():
        fig, axs = plt.subplots(1, len(gwf_sims), 
                    figsize=(4 * len(gwf_sims), 4)
                )
        fig.suptitle(f"Head - flow angle 45")

        for idx, (grid, sim) in enumerate(gwf_sims.items()):
                plot_flow(sim, axs[idx])

        pad = 5 # in points
        for ax, col in zip(axs, gwf_sims.keys()):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')     

    plt.show()

def plot_flow(sim, ax):
    gwf = sim.get_model()
    head = gwf.output.head().get_data()
    bud = gwf.output.budget()
    spdis = bud.get_data(text="DATA-SPDIS")[0]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
    spdis, gwf
)
    vmin = head.min()
    vmax = head.max()
     
    pmv = flopy.plot.PlotMapView(gwf, ax=ax)
    pmv.plot_grid(colors="k", alpha=0.1)
    pc = pmv.plot_array(
        head, vmin=vmin, vmax=vmax, alpha=0.5
    )
    pmv.plot_vector(qx, qy, color="white")
    plt.colorbar(pc)

    ax.set_xlabel("x[m]")
    ax.set_ylabel("y[m]")
    ax.set_aspect("equal")

def plot_concentrations(gwf_sims):
    for wave_func in wave_functions:   
        with styles.USGSPlot():     
            fig, axs = plt.subplots(
                nrows=len(schemes), 
                ncols=len(grids), 
                figsize=(4 * len(grids) + 4, 4 * len(schemes))
            )
            fig.suptitle(f"Concentration - {wave_func}")

            for idx_scheme, scheme in enumerate(schemes):
                for idx_grid, grid in enumerate(grids):
                    sim = gwf_sims[(grid, scheme, wave_func)]
                    plot_concentration(sim, axs[idx_scheme, idx_grid])

            pad = 5 # in points
            for ax, col in zip(axs[0], grids):
                ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

            for ax, row in zip(axs[:,0], schemes):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
            
            fig.subplots_adjust(left=0.25, top=0.95)

        plt.show()

def plot_concentration(sim, ax):
    gwt = sim.get_model()
    ucnobj_mf6 = gwt.output.concentration()
    conc = ucnobj_mf6.get_data(totim=total_time).flatten()

    vmin = conc.min()
    vmax = conc.max()
   
    pmv = flopy.plot.PlotMapView(gwt, ax=ax)
    pc = pmv.plot_array(
        conc, vmin=vmin, vmax=vmax, alpha=0.5
    )
    plt.colorbar(pc)

    ax.set_xlabel("x[m]")
    ax.set_ylabel("y[m]")
    ax.set_aspect("equal")

def plot_concentration_cross_sections(gwf_sims):
    with styles.USGSPlot():     
        fig, axs = plt.subplots(
            nrows=len(wave_functions), 
            ncols=len(grids), 
            figsize=(4 * len(grids) + 4, 4 * len(wave_functions))
        )
        fig.suptitle(f"Concentration cross-section")

        for wave_idx, wave_func in enumerate(wave_functions):
            for grid_idx, grid in enumerate(grids):
                ax = axs[wave_idx][grid_idx]
                plot_concentration_analytical(wave_func, ax)
                for scheme in schemes:
                    sim = gwf_sims[(grid, scheme, wave_func)]
                    plot_concentration_cross_section(sim, scheme, wave_func, ax)

                ax.legend()
                ax.set_xlabel("x[m]")
                ax.set_ylabel("C[-]")
    
        pad = 5 # in points
        for ax, col in zip(axs[0], grids):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

        for ax, row in zip(axs[:,0], wave_functions):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

        fig.subplots_adjust(left=0.25, top=0.95)

    plt.show()

def plot_concentration_analytical(analytical_func, ax):
    x = np.linspace(0, Length, 100)
    y = Width / 2

    conc = exact_solution_concentration(x, y, analytical_func)

    ax.plot(
            x,
            conc,
            linestyle='-',
            mfc="none",
            markersize="4",
            label = "exact"
        )

def plot_concentration_cross_section(sim, scheme, analytical_func, ax):
    gwt = sim.get_model()
    ucnobj_mf6 = gwt.output.concentration()
    conc = ucnobj_mf6.get_data(totim=total_time).flatten()

    grid = gwt.modelgrid
    xc = grid.xcellcenters
    yc = grid.ycellcenters

    ix = GridIntersect(grid)
    x_line = LineString([(0, Width / 2), (Length, Width / 2)])
    xi = ix.intersect(x_line)
    x_along_line = sorted(xc[xi.cellids.tolist()])

    interp = LinearNDInterpolator(list(zip(xc, yc)), conc)
    conc_interp = interp([(i, Width / 2) for i in x_along_line])

    ax.plot(
        x_along_line, 
        conc_interp,
        linestyle='--',
        marker="o",
        mfc="none",
        markersize="4",
        label = scheme
        )

    

# %% [markdown]
# # Running the example
#
# Define and invoke a function to run the example scenario, then plot results.

# %%
def scenario(silent=True):
    gwf_sims = {}
   
    # Build and run the gwt models on different grids
    for grid in grids:
        gwf_sims[grid] = build_mf6gwf(grid)

    # Build and run gwf with sin^2 wave on all grids
    gwt_sims = {}
    combinations = list(itertools.product(*[grids, schemes, wave_functions]))
    for combination in combinations:
        gwt_sims[combination] = build_mf6gwt(combination[0],  combination[1], combination[2])
    
    if write:
        print("\n writting flow models:")
        for grid, sim in gwf_sims.items():
            print(f"- {grid} grid")
            write_models(sim, silent=silent)
        
        print("\n writting transport models:")
        for key, sim in gwt_sims.items():
            grid, scheme, wave = key
            print(f"- {grid} grid \t {scheme} scheme \t {wave}-function")
            write_models(sim, silent=silent)

    if run:
        print("\n Running flow models:")
        for grid, sim in gwf_sims.items():
            print(f"- {grid} grid")
            run_models(sim, silent=silent)

        print("\n Running transport models:")
        for key, sim in gwt_sims.items():
            grid, scheme, wave = key
            print(f"- {grid} grid \t {scheme} scheme \t {wave}-function")
            run_models(sim, silent=silent)
    if plot:
        print("\n Plotting results")
        plot_results(gwf_sims, gwt_sims)


scenario()
