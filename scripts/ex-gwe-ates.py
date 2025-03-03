# ## Aquifer thermal energy storage (ATES)
#
# Application of a MODFLOW 6 groundwater energy transport (GWE) model to
# simulate a aquifer thermal energy storage (ATES) for storing (or
# extracting) energy from an aquifer over an annually repeating period.
#
# This model uses a DISV grid type.  The flow system consists of three
# layers of marginally porous material, where the upper and lower layers
# share the same hydraulic and thermal properties. Water is injected and
# extracted into the middle layer sandwiched between the upper and lower
# layers.  Hydraulic and thermal properties in the middle layer are distinct
# from the upeper and lower layers.
#
# Energy is loaded into the left side of the domain by specifying the
# temperature of the injected water.  Water is extracted from the aquifer
# at its calculated temperature.


# ### Initial setup
#
# Import dependencies, define the example name and workspace, and read settings from environment variables.

# +
from pathlib import Path
from pprint import pformat

import flopy
import git
import matplotlib.pyplot as plt
import numpy as np
import pooch
from flopy.mf6 import MFSimulation
from modflow_devtools.misc import get_env, timed

# Example name and workspace paths. If this example is running
# in the git repository, use the folder structure described in
# the README. Otherwise just use the current working directory.
sim_name = "ex-gwe-ates"
gwfname = "gwf-" + sim_name.split("-")[-1]
gwename = "gwe-" + sim_name.split("-")[-1]

try:
    root = Path(git.Repo(".", search_parent_directories=True).working_dir)
except:
    root = None

workspace = root / "examples" if root else Path.cwd()
figs_path = root / "figures" if root else Path.cwd()
data_path = root / "data" / sim_name if root else Path.cwd()
sim_ws = workspace / sim_name

# Settings from environment variables
write = get_env("WRITE", True)
run = get_env("RUN", True)
plot = get_env("PLOT", True)
plot_show = get_env("PLOT_SHOW", True)
plot_save = get_env("PLOT_SAVE", True)
gif_save = get_env("GIF", False)
# -

# ### Define parameters
#
# Define model units, parameters and other settings.

# +
# Model units
length_units = "meters"
time_units = "days"

# Model parameters
nper = 127  # Number of stress periods
nlay = 1  # Number of layers
k11_zn1 = 8.64  # Zone 1 horizontal hydraulic conductivity ($m/d$)
k11_zn2 = 0.000864  # Zone 2 horizontal hydraulic conductivity ($m/d$)
icelltype = 0  # Saturated thickness will be held constant
strt = 1000.0  # Starting head ($m$)
ss = 1e-3  # Specific storage ($-$)
sy_zn1 = 0.01  # Zone 1 specific yield ($-$)
sy_zn2 = 0.10  # Zone 2 specific yield ($-$)
prsity_zn1 = 0.01  # Zone 1 porosity ($-$)
prsity_zn2 = 0.10  # Zone 2 porosity ($-$)
strt_temp = 50.0  # Starting temperature of aquifer ($^{\circ}C$)
t_inj = 20  # Temperature of injected water ($^{\circ}C$)
al = 0.1  # Longitudinal dispersivity ($m$)
ath1 = 0.01  # Transverse dispersivity ($m$)
kts_zn1 = 3.0  # Zone 1 thermal conductivity ($\frac{W}{m \cdot ^{\circ} C}$)
kts_zn2 = 1.4  # Zone 2 thermal conductivity ($\frac{W}{m \cdot ^{\circ} C}$)
ktw = 0.58  # Thermal conductivity of water ($\frac{W}{m \cdot ^{\circ} C}$)
cps_zn1 = 1100.0  # Zone 1 heat capacity ($\frac{J}{kg \cdot ^{\circ} C}$)
cps_zn2 = 850.0  # Zone 2 heat capacity ($\frac{J}{kg \cdot ^{\circ} C}$)
cpw = 4180.0  # Heat capacity of water ($\frac{J}{kg \cdot ^{\circ} C}$)
rhow = 1000.0  # Density of water ($kg/m^3$)
rhos = 2500.0  # Density of dry solid aquifer material ($kg/m^3$)
lhv = 2500.0  # Latent heat of vaporization ($kJ/kg$)
scheme = "TVD"  # Advection solution scheme ($-$)

# Stress period pumping
adj_factor = 0.01  # well pumping adjustment factor
welq = [
    0.0,
    0.0,
    -9.6,
    -19.3,
    -28.9,
    -38.6,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -38.6,
    -28.9,
    -19.3,
    -9.6,
    0.0,
    0.0,
    0.0,
    0.0,
    9.6,
    19.3,
    28.9,
    38.6,
    48.2,
    48.2,
    48.2,
    48.2,
    48.2,
    48.2,
    38.6,
    28.9,
    19.3,
    9.6,
    0.0,
    0.0,
    0.0,
    0.0,
    -9.6,
    -19.3,
    -28.9,
    -38.6,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -38.6,
    -28.9,
    -19.3,
    -9.6,
    0.0,
    0.0,
    0.0,
    0.0,
    9.6,
    19.3,
    28.9,
    38.6,
    48.2,
    48.2,
    48.2,
    48.2,
    48.2,
    48.2,
    38.6,
    28.9,
    19.3,
    9.6,
    0.0,
    0.0,
    0.0,
    0.0,
    -9.6,
    -19.3,
    -28.9,
    -38.6,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -38.6,
    -28.9,
    -19.3,
    -9.6,
    0.0,
    0.0,
    0.0,
    0.0,
    9.6,
    19.3,
    28.9,
    38.6,
    48.2,
    48.2,
    48.2,
    48.2,
    48.2,
    48.2,
    38.6,
    28.9,
    19.3,
    9.6,
    0.0,
    0.0,
    0.0,
    0.0,
    -9.6,
    -19.3,
    -28.9,
    -38.6,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -48.2,
    -38.6,
    -28.9,
    -19.3,
    -9.6,
    0.0,
    0.0,
    0.0,
]

# Load a file stored in the data directory for building out the DISV grid
fname = "disv_nodes.fem"
fpath = pooch.retrieve(
    url=f"https://github.com/MODFLOW-ORG/modflow6-examples/raw/develop/data/{sim_name}/{fname}",
    fname=fname,
    path=data_path,
    known_hash="md5:d107d2a5e01646a861e73bb3465f0747",
)
# fpath = data_path / fname


# Model timing
#
nper = len(welq)
perlen = [10] * nper
nstp = [1] * nper
tsmult = [1.0] * nper
unitconv = 86400

# Solver parameters
nouter, ninner = 200, 300
hclose, rclose, relax = 1e-7, 1e-3, 0.97
transient = {0: True}

# -


# +
# ### Functions for generating the model grid and identifying which zone the cells fall into
def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


def read_dims(flname):
    with open(flname, "r") as f:
        for line in f:
            if "DIMENS" in line:
                line = next(f)
                m_arr = line.strip().split()
                nnode = int(m_arr[0])
                nelement = int(m_arr[1])
                break

    return nnode, nelement


def read_nodes(flname, nelement):
    iverts = []
    with open(flname, "r") as f:
        for line in f:
            if "NODE" in line:
                for i in np.arange(nelement):
                    line = next(f)
                    m_arr = line.strip().split()
                    iverts.append([i] + [int(itm) - 1 for itm in m_arr])

                break

    return iverts


def read_coords(flname):
    all_vals = []
    with open(flname, "r") as f:
        for line in f:
            if "COOR" in line:
                line = next(f)
                while "GK_COOR" not in line:
                    m_arr = line.strip().split(",")
                    for val in m_arr:
                        if not val == "":
                            all_vals.append(float(val))

                    line = next(f)

                break

    return all_vals


def process_verts(iverts, verts):
    xc, yc = [], []
    xyverts = []
    for iv in iverts:
        xv, yv = [], []
        for v in iv[1:]:
            tiv, txv, tyv = verts[v]
            xv.append(txv)
            yv.append(tyv)

        xc.append(np.mean(xv))
        yc.append(np.mean(yv))
        xyverts.append(list(zip(xv, yv)))

    return xc, yc, xyverts


def get_bnd_inflow_locs(verts):
    left_bnd_verts = []

    for itm in verts:
        xcoord = itm[1]
        ycoord = itm[2]

        if ycoord >= 15 and ycoord <= 35:
            if xcoord < 0.5:
                left_bnd_verts.append(itm)

    return left_bnd_verts


def generate_bnd_features(verts, iverts, left_bnd_verts):
    # Store the ids of the new features
    inQ_feat = []

    for i in np.arange(len(left_bnd_verts) - 1):
        pt1 = left_bnd_verts[i]
        pt1_id = pt1[0]
        pt1_y = pt1[2]
        pt2 = left_bnd_verts[i + 1]
        pt2_id = pt2[0]
        pt2_y = pt2[2]
        if i == 0:
            newpt3 = [len(verts), 0.0, pt2_y]
            newpt4 = [len(verts) + 1, 0.0, pt1_y]

            # Store the vertices
            verts.append(newpt3)
            verts.append(newpt4)
        else:
            newpt3 = [len(verts), 0.0, pt2_y]
            prev_pt3 = newpt3

            # Store the vertex
            verts.append(newpt3)

        # add the new ivert to list iverts
        if i == 0:
            new_ivert = [len(iverts), pt1_id, pt2_id, newpt3[0], newpt4[0]]
        else:
            new_ivert = [len(iverts), pt1_id, pt2_id, newpt3[0], prev_pt3[0]]

        iverts.append(new_ivert)
        inQ_feat.append(new_ivert)

        # pt 3 will become pt4 in the next feature
        prev_pt3 = newpt3

    # Return
    return verts, iverts, inQ_feat


def create_cell2d(iverts, xyverts, xc, yc):
    cell2d = []
    for ix, iv in enumerate(iverts):
        xv, yv = np.array(xyverts[ix]).T
        if flopy.utils.geometry.is_clockwise(xv, yv):
            rec = [iv[0], xc[ix], yc[ix], len(iv[1:])] + iv[1:]
        else:
            iiv = iv[1:][::-1]
            rec = [iv[0], xc[ix], yc[ix], len(iiv)] + iiv

        cell2d.append(rec)

    return cell2d


def read_finite_element_mesh(flname):
    # Read dimensions
    nnode, nelement = read_dims(flname)

    # Read in vertices
    iverts = read_nodes(flname, nelement)

    # Read in continuous list of FEFLOW node coordinates
    # (assumes all x come first, then all y)
    all_vals = read_coords(flname)

    # Stitch coord locations together
    all_x, all_y = split_list(all_vals)
    verts = []
    for i, (x, y) in enumerate(zip(all_x, all_y)):
        verts.append([int(i), float(x), float(y)])

    # Create rectangular this boundary cells where specified inflows will enter
    left_bnd_verts = get_bnd_inflow_locs(verts)
    # sort, upper to lower
    left_bnd_verts.sort(key=lambda x: x[2], reverse=True)
    # generate new 2d DISV objects for inflow
    verts, iverts, inQ_iverts = generate_bnd_features(verts, iverts, left_bnd_verts)

    # Calculate cell center locations for each element
    # and store xyverts for later use.
    xc, yc, xyverts = process_verts(iverts, verts)

    # Finally create a cell2d record
    cell2d = create_cell2d(iverts, xyverts, xc, yc)

    # Return disv objects
    return verts, cell2d, inQ_iverts


def determine_zone(cell2d):
    low_k_id = []
    high_k_id = []
    for itm in cell2d:
        id = itm[0]
        y_coord = itm[2]
        if y_coord >= 35.0 or y_coord <= 15.00:
            low_k_id.append(id)
        else:
            high_k_id.append(id)

    low_k_id.sort()
    high_k_id.sort()
    return low_k_id, high_k_id


def determine_bnd(cell2d, verts):
    left_bnd = []
    right_bnd = []
    for itm in cell2d:
        id = itm[0]
        incld_verts = itm[-3:]

        xlct = 0
        xrct = 0
        for vert in incld_verts:
            x = verts[vert][1]
            y = verts[vert][2]
            if y >= 15 and y <= 35:
                if x < 0.25:
                    xlct += 1
                    if xlct == 2:
                        left_bnd.append(id)

                elif x > 135.0 - 0.01:
                    xrct += 1
                    if xrct == 2:
                        right_bnd.append(id)

    return left_bnd, right_bnd


def determine_param(low_k_id, high_k_id, mode):
    msg0 = "Missing an index"
    msg1 = "Missing ID is: "
    p_arr = []

    if mode.lower() == "k":
        val_zn1 = k11_zn1
        val_zn2 = k11_zn2
    elif mode.lower() == "sy":
        val_zn1 = sy_zn1
        val_zn2 = sy_zn2
    elif mode.lower() == "kts":
        val_zn1 = kts_zn1 * unitconv
        val_zn2 = kts_zn2 * unitconv
    elif mode.lower() == "prsity":
        val_zn1 = prsity_zn1
        val_zn2 = prsity_zn2
    elif mode.lower() == "cps":
        val_zn1 = cps_zn1
        val_zn2 = cps_zn2

    id_max = np.max([np.max(low_k_id), np.max(high_k_id)])
    assert len(low_k_id + high_k_id) == id_max + 1, msg0
    for idx in np.arange(id_max + 1):
        if idx in low_k_id:
            p_arr.append(val_zn2)
        elif idx in high_k_id:
            p_arr.append(val_zn1)
        else:
            assert False, msg1 + str(idx) + " (0-based)"

    return p_arr


# -


# +
def build_model(sim_name, verts, cell2d, top, botm):
    name = sim_name

    # build MODFLOW 6 files
    sim = flopy.mf6.MFSimulation(
        sim_name=name, version="mf6", exe_name="mf6", sim_ws=sim_ws
    )

    # create tdis package
    tdis_rc = []
    for i in range(nper):
        tdis_rc.append((perlen[i], nstp[i], tsmult[i]))

    flopy.mf6.ModflowTdis(sim, time_units=time_units, nper=nper, perioddata=tdis_rc)

    # create gwf model
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        save_flows=True,
    )

    # create iterative model solution and register the gwf model with it
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="DBD",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename=f"{gwfname}.ims",
    )
    sim.register_ims_package(ims, [gwf.name])

    # Instantiating MODFLOW 6 discretization package
    flopy.mf6.ModflowGwfdisv(
        gwf,
        length_units=length_units,
        nogrb=True,
        ncpl=len(cell2d),
        nvert=len(verts),
        nlay=nlay,
        top=top,
        botm=botm,
        idomain=1,
        vertices=verts,
        cell2d=cell2d,
        pname="DISV",
        filename=f"{gwfname}.disv",
    )

    # Instantiating MODFLOW 6 node property flow package
    # Determine which K zone each element falls in:
    low_k_id, high_k_id = determine_zone(cell2d)

    # Set 1D K list based on results of previous function
    k11 = determine_param(low_k_id, high_k_id, "k")

    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=icelltype,
        # xt3doptions="XT3D",
        k=k11,
        save_specific_discharge=True,
        save_saturation=True,
        pname="NPF",
        filename=f"{gwfname}.npf",
    )

    # Instantiating MODFLOW 6 initial conditions package
    flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # Instantiating MODFLOW 6 storage package
    sy = determine_param(low_k_id, high_k_id, "sy")
    flopy.mf6.ModflowGwfsto(
        gwf,
        ss=ss,
        sy=sy,
        transient={0: True},
        pname="STO",
        filename=f"{gwfname}.sto",
    )

    # Instantiate WEL package
    # Determine which CVs are on the right and left boundary
    left_bnd, right_bnd = determine_bnd(cell2d, verts)
    wel_spd = {}
    for tm, q in enumerate(welq):
        wel_q = []
        for cellid in left_bnd:
            if q > 0.0:
                q_temp = t_inj
            else:
                q_temp = 0.0

            q_adj = q * adj_factor
            wel_q.append([(0, cellid), q_adj, q_temp])

        wel_spd.update({tm: wel_q})

    flopy.mf6.ModflowGwfwel(
        gwf,
        auxiliary="TEMPERATURE",
        save_flows=True,
        stress_period_data=wel_spd,
        pname="WEL",
        filename=f"{gwfname}.wel",
    )

    # Instantiating MODFLOW 6 output control package (flow model)
    head_filerecord = f"{gwfname}.hds"
    budget_filerecord = f"{gwfname}.cbc"
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        saverecord=[("HEAD", "All"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    # -------------------
    # GWE Model
    # -------------------

    # Instantiating MODFLOW 6 groundwater transport model
    gwe = flopy.mf6.MFModel(
        sim,
        model_type="gwe6",
        modelname=gwename,
        model_nam_file=f"{gwename}.nam",
    )

    # Create iterative model solution and register the gwe model with it
    imsgwe = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="SIMPLE",
        no_ptcrecord="all",
        linear_acceleration="bicgstab",
        scaling_method="NONE",
        reordering_method="NONE",
        outer_maximum=nouter,
        outer_dvclose=hclose * 1000,
        under_relaxation="dbd",
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.08,
        under_relaxation_gamma=0.05,
        under_relaxation_momentum=0.0,
        backtracking_number=20,
        backtracking_tolerance=2.0,
        backtracking_reduction_factor=0.2,
        backtracking_residual_limit=5.0e-4,
        inner_maximum=ninner,
        inner_dvclose=hclose * 1000,
        relaxation_factor=0.0,
        number_orthogonalizations=2,
        preconditioner_levels=8,
        preconditioner_drop_tolerance=0.001,
        rcloserecord=f"{rclose} strict",
        filename=f"{gwename}.ims",
    )
    sim.register_ims_package(imsgwe, [gwe.name])

    # Instantiating MODFLOW 6 heat transport discretization package
    flopy.mf6.ModflowGwedisv(
        gwe,
        nogrb=True,
        nlay=nlay,
        ncpl=len(cell2d),
        nvert=len(verts),
        top=top,
        botm=botm,
        idomain=1,
        vertices=verts,
        cell2d=cell2d,
        pname="DISV-GWE",
        filename=f"{gwename}.disv",
    )

    # Instantiating MODFLOW 6 heat transport initial temperature
    flopy.mf6.ModflowGweic(gwe, strt=strt_temp, pname="IC", filename=f"{gwename}.ic")

    # Instantiating MODFLOW 6 heat transport advection package
    flopy.mf6.ModflowGweadv(gwe, scheme=scheme, pname="ADV", filename=f"{gwename}.adv")

    # Instantiating MODFLOW 6 heat transport energy storage package (consider renaming to est)
    prsity = determine_param(low_k_id, high_k_id, "prsity")
    cps = determine_param(low_k_id, high_k_id, "cps")
    flopy.mf6.ModflowGweest(
        gwe,
        porosity=prsity,
        heat_capacity_water=cpw,
        density_water=rhow,
        latent_heat_vaporization=lhv,
        heat_capacity_solid=cps,
        density_solid=rhos,
        pname="EST",
        filename=f"{gwename}.est",
    )

    # Instantiating MODFLOW 6 heat transport dispersion package
    kts = determine_param(low_k_id, high_k_id, "kts")
    flopy.mf6.ModflowGwecnd(
        gwe,
        xt3d_off=True,
        alh=al,
        ath1=ath1,
        ktw=ktw,
        kts=kts,
        pname="CND",
        filename=f"{gwename}.cnd",
    )

    # Instantiating MODFLOW 6 source/sink mixing package for dealing with
    # auxiliary temperature specified in WEL boundary package.
    sourcerecarray = [
        ("WEL", "AUX", "TEMPERATURE"),
    ]
    flopy.mf6.ModflowGwessm(
        gwe, sources=sourcerecarray, pname="SSM", filename=f"{gwename}.ssm"
    )

    # Instantiating MODFLOW 6 heat transport output control package
    flopy.mf6.ModflowGweoc(
        gwe,
        budget_filerecord=f"{gwename}.cbc",
        temperature_filerecord=f"{gwename}.ucn",
        temperatureprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("TEMPERATURE", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("TEMPERATURE", "ALL"), ("BUDGET", "ALL")],
        filename=f"{gwename}.oc",
    )

    # Instantiate Gwf-Gwe Exchange package
    flopy.mf6.ModflowGwfgwe(
        sim,
        exgtype="GWF6-GWE6",
        exgmnamea=gwfname,
        exgmnameb=gwename,
        filename=f"{gwename}.gwfgwe",
    )

    return sim


# -


# +
def write_model(sim, silent=True):
    print("Writing input for ATES test model...")
    success = False
    if isinstance(sim, MFSimulation):
        sim.write_simulation(silent=silent)
    else:
        sim.write_input()


@timed
def run_model(sim, silent=True):
    print("running ATES test model...")
    if isinstance(sim, MFSimulation):
        success, buff = sim.run_simulation(silent=silent, report=True)
    else:
        sim.run_model(silent=silent, report=True)
    assert success, pformat(buff)


# -

# ### Plotting results
#
# Define functions to plot model results.

# +
# Figure properties
figure_size_bc = (6, 3)


def plot_temperature(sim, idx):
    print("plotting select results...")

    gwf = sim.get_model(gwfname)
    gwe = sim.get_model(gwename)

    # get specific discharge
    bdobj = gwf.output.budget()
    spdis = bdobj.get_data(text="DATA-SPDIS")[22]
    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    # get binary output
    temps = gwe.output.temperature().get_alldata()
    vmin = temps.min()
    vmax = temps.max()
    times = gwe.output.temperature().get_times()

    # make bc figure
    fig = plt.figure(figsize=figure_size_bc)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(model=gwe, ax=ax)
    pmv.plot_grid(alpha=0.25, linewidth=0.5)
    cb = pmv.plot_array(gwe.est.porosity.array, cmap="Set2")
    ax.set_xlabel("X, m")
    ax.set_ylabel("Y, m")
    # ax.set_title(f"Porosity View")
    ax.text(15, 42.5, "Zone 2", ha="center")
    ax.text(15, 25, "Zone 1", ha="center")
    ax.text(15, 7.5, "Zone 2", ha="center")
    plt.tight_layout()
    if plot_show:
        plt.show()
    if plot_save:
        fpth = figs_path / f"{sim_name}-prsity.png"
        fig.savefig(fpth, dpi=600)

    plt.close("all")

    # pumping stress
    welx = [x * 10 for x in range(1, len(welq) + 1)]
    welq_tot = [q * gwf.wel.maxbound.array * adj_factor for q in welq]

    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(welx, welq_tot, linewidth=0.25)
    plt.axhline(y=0.0, color="k", linewidth=1, linestyle="-")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    ax.set_xlabel("Time, $days$", fontsize=7)
    ax.set_ylabel(r"Injection/extraction rate, $\dfrac{m^3}{day}$", fontsize=7)
    plt.tight_layout()
    if plot_save:
        fpth = figs_path / f"{sim_name}-pmprate.png"
        fig.savefig(fpth, dpi=300)

    # make results plot
    fig, axs = plt.subplots(2, 2)

    # plot times in the original publication
    plot_periods = [
        20,  # after 1 injection stress period
        33,  # after first block of 13 injection periods
        51,  # end of 1st extraction period
        -1,  # end of simulation
    ]

    iplot = -1
    for row in range(2):
        for col in range(2):
            iplot += 1
            sp_idx = plot_periods[iplot]
            temp = temps[sp_idx]

            ax = axs[row, col]
            pmv = flopy.plot.PlotMapView(model=gwe, ax=ax)
            pmv.plot_grid(alpha=0.25, linewidth=0.5)
            cb = pmv.plot_array(temp, alpha=0.7, cmap="jet", vmin=vmin, vmax=vmax)
            if iplot in [2, 3]:
                ax.set_xlabel("X position, m", fontsize=7)
            if iplot in [0, 2]:
                ax.set_ylabel("Y position, m", fontsize=7)

            ax.tick_params(axis="both", labelsize=7)
            letter = chr(ord("@") + iplot + 1)
            ax.text(3, 44, letter, fontsize=10, fontweight="bold")

    plt.subplots_adjust(right=0.9)
    cb_ax = plt.axes((0.925, 0.25, 0.025, 0.5))
    clb = plt.colorbar(cb, cax=cb_ax, shrink=0.35)
    clb.ax.set_yticklabels(
        [
            r"20$^{\circ}C$",
            r"20$^{\circ}C$",
            r"25$^{\circ}C$",
            r"30$^{\circ}C$",
            r"35$^{\circ}C$",
            r"40$^{\circ}C$",
            r"45$^{\circ}C$",
            r"50$^{\circ}C$",
        ]
    )
    clb.ax.tick_params(labelsize=7)
    fig.set_size_inches(8, 4)

    if plot_show:
        plt.show()
    if plot_save:
        fpth = figs_path / f"{sim_name}-temp2x2.png"
        fig.savefig(fpth, dpi=600)


def make_animated_gif(sim):
    print("generating animation...")
    from matplotlib.animation import FuncAnimation, PillowWriter

    gwe = sim.get_model(gwename)

    # get binary output
    temps = gwe.output.temperature().get_alldata()
    vmin = temps.min()
    vmax = temps.max()
    times = gwe.output.temperature().get_times()

    # make base figure
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pmv = flopy.plot.PlotMapView(model=gwe, ax=ax)
    pmv.plot_grid(alpha=0.25)
    tempmesh = pmv.plot_array(temps[0], alpha=0.7, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(
        tempmesh,
        shrink=0.85,
        ax=ax,
        label="Temperature",
        location="bottom",
        fraction=0.1,
    )

    def init():
        ax.set_xlabel("X, m")
        ax.set_ylabel("Y, m")
        # ax.set_aspect("equal")
        ax.set_title(f"Time = {times[0]} days")

    def update(i):
        tempmesh.set_array(temps[i].flatten())
        ax.set_title(f"Time = {times[i]} days")

    ani = FuncAnimation(fig, update, range(1, len(times)), init_func=init)
    # interval=25,
    writer = PillowWriter(fps=10)
    fpth = figs_path / f"{sim_name}.gif"
    ani.save(fpth, writer=writer)


def plot_results(sim, idx):
    plot_temperature(sim, idx)
    if plot_save and gif_save:
        make_animated_gif(sim)


# -


# +
# ### Running the example
#
# Define a function to run the example scenarios and plot results.
def scenario(idx, silent=True):
    verts, cell2d, inQ_iverts = read_finite_element_mesh(fpath)
    top = np.ones((len(cell2d),))
    botm = np.zeros((1, len(cell2d)))
    modelgrid = flopy.discretization.VertexGrid(verts, cell2d, top=top, botm=botm)

    sim = build_model(sim_name, verts, cell2d, top, botm)
    if isinstance(sim, MFSimulation) and write:
        write_model(sim, silent=silent)
    if run:
        run_model(sim, silent=silent)
    if plot:
        plot_results(sim, idx)


# -

# +
# Initiate model construction, write, run, and plot sequence
scenario(0, silent=True)
# -
