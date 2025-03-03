# ## Infiltrating Heat Front
#
# An example demonstrating that a specified energy flux boundary condition
# works as expected when compared to an analytical solution.
#

# ### Initial Setup
#
# Import dependencies, define the example name and workspace,
# and read settings from environment variables.

# +
import math
from pathlib import Path
from pprint import pformat

import flopy
import git
import matplotlib.pyplot as plt
import numpy as np
from flopy.mf6 import MFSimulation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from modflow_devtools.misc import get_env, timed

# -

# +
# Example name and workspace paths. If this example is running
# in the git repository, use the folder structure described in
# the README. Otherwise just use the current working directory.
sim_name = "ex-gwe-danckwerts"
# shorten model names so they fit in 16-char limit
gwfname = "gwf-" + sim_name.replace("ex-gwe-", "")
gwename = "gwe-" + sim_name.replace("ex-gwe-", "")

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
# -


# +
# ### Define analytical solution functions
# Analytical solution derived by Alden Provost, similar in form to
# Barends (2010) Equation 5 - but remember that that solution
# pins the temperature of the top cell to a specified temperature
# rather than specifying what the input energy flux is.
def flux_analyt(t, z, qt0, qtinfil, v, d):
    if t == 0.0:
        flux = qt0
    else:
        denom = 2.0 * math.sqrt(d * t)
        ztermm = (z - v * t) / denom
        ztermp = (z + v * t) / denom
        vterm = v * z / d
        if vterm < 100.0:
            # might need to adjust this limit
            flux = qt0 + (qtinfil - qt0) * 0.5 * (
                math.erfc(ztermm) + math.exp(vterm) * math.erfc(ztermp)
            )
        else:
            zeta = 1.0 / (1.0 + 0.47047 * ztermp)
            polyterm = zeta * (0.3480242 + zeta * (-0.0958798 + zeta * 0.7478556))
            flux = qt0 + 0.5 * (qtinfil - qt0) * (
                math.erfc(ztermm) + math.exp(vterm - ztermp**2) * polyterm
            )

    return flux


# -


# +
def temp_analyt(t, z, t0, tinfil, v, d):
    if t == 0.0:
        temp = t0
    else:
        denom = 2.0 * math.sqrt(d * t)
        ztermm = (z - v * t) / denom
        ztermp = (z + v * t) / denom
        vterm = v * z / d
        if vterm < 100.0:
            # might need to adjust this limit
            temp = t0 + 0.5 * (tinfil - t0) * (
                math.erfc(ztermm) + math.exp(vterm) * math.erfc(ztermp)
            )
        else:
            zeta = 1.0 / (1.0 + 0.47047 * ztermp)
            polyterm = zeta * (0.3480242 + zeta * (-0.0958798 + zeta * 0.7478556))
            temp = t0 + 0.5 * (tinfil - t0) * (
                math.erfc(ztermm) + math.exp(vterm - ztermp**2) * polyterm
            )

    return temp


# -

# +
# ### Define parameters
#
# Define model units, parameters and other settings.

# Model units
length_units = "meters"
time_units = "days"

# Model parameters
nlay = 101  # Number of layers ($-$)
nrow = 1  # Number of rows ($-$)
ncol = 1  # Number of columns ($-$)
nper = 2  # Number of simulated periods ($-$)
delr = 1.0  # Cell width ($m$)
delc = 1.0  # Cell length ($m$)
delz = 0.1  # Cell thickness ($m$)
strt = 0.05  # Initial head ($m$)
top = 10.00  # Top of the model grid ($m$)
strt_temp = 10.0  # Initial temperature ($^{\circ}C$)
scheme = "Upstream"  # Advection scheme ($-$)
dispersivity = 0.0  # Longitudinal mechanical dispersion term ($m$)
prsity = 0.2  # Porosity ($-$)
rhos = 1500.0  # Density of dry solid aquifer material ($\frac{kg}{m^3}$)
cps = 760.0  # Heat capacity of dry solid aquifer material ($\frac{J}{kg \cdot ^{\circ} C}$)
rhow = 1000.0  # Density of water ($\frac{kg}{m^3}$)
cpw = 4183.0  # Heat capacity of water ($\frac{J}{kg \cdot ^{\circ} C}$)
ktw = 0.5918  # Thermal conductivity of water ($\frac{W}{m \cdot ^{\circ} C}$)
kts = 0.27  # Thermal conductivity of solid aquifer material ($\frac{W}{m \cdot ^{\circ} C}$)
finf = 0.01  # Infiltration rate ($\frac{m}{d}$)
chdval = 0.05  # Constant head at the model outlet ($m$)
thtr = 0.0001  # Residual water content of the unsaturated zone ($-$)
thts = 0.20  # Saturated water content of the unsaturated zone ($-$)
thti = 0.055  # Initial water content of the unsaturated zone ($-$)
eps = 4.0  # Brooks-Corey epsilon parameter ($-$)
vks = 1.0  # Vertical hydraulic conductivity of the unsaturated zone ($\frac{m}{d}$)
t0 = 10.0  # Initial temperature in simulation domain ($^{\circ} C$)
tinfil = 20.0  # Temperature of infiltrating water ($^{\circ} C$)

# -

# +
# Solver parameters
nouter, ninner = 100, 300
hclose, rclose, relax = 1e-9, 1e-3, 0.97
steady = {0: False, 1: False}
transient = {0: True, 1: True}

# -

# +
# Values that do not show up in the table generated for latex
perlen = [1.0e9, 100.0]  # Length of stress periods ($-$)
nstp = [1, 100]  # Number of time steps per stress period ($-$)
tsmult = len(perlen) * [1.0]  # Time step multiplier ($-$)

top += 0.0005  # Add thin amount to upper layer for placing the node at 10.0 m
botm = [9.9995]  # Thickness of uppermost boundary cell ($m$)
for i in np.arange(1, nlay):
    bot = 10.0 - (i * delz)
    botm.append(round(bot, 1))

idomain = [1] * nlay

q = finf  # infiltration rate
area = delr * delc
rhowCpw = cpw * rhow
rhosCps = cps * rhos
lhv = 2500.0  # Latent heat of vaporization

# transient uzf info
# iuzno  cellid landflg ivertcn surfdp vks thtr thts thti eps [bndnm]
uzf_pkdat = [[0, (0, 0, 0), 1, 1, 0.00001, vks, thtr, thts, thti, eps]]

# Continue building the UZF list of objects
for iuzno in np.arange(1, 100, 1):
    if iuzno < nlay - 2:
        ivertconn = iuzno + 1
    else:
        ivertconn = -1

    uzf_pkdat.append(
        [iuzno, (iuzno, 0, 0), 0, ivertconn, 0.01, vks, thtr, thts, thti, eps]
    )

iuz_cell_dict = {}
cell_iuz_dict = {}
for i, itm in enumerate(uzf_pkdat):
    iuz_cell_dict.update({itm[0]: (itm[1][0], itm[1][1], itm[1][2])})
    cell_iuz_dict.update({(itm[1][0], itm[1][1], itm[1][2]): itm[0]})

extdp = 0.0
pet = 0.0
extwc = 0.0
zero = 0.0
uzf_spd = {
    0: [[0, finf, pet, extdp, extwc, zero, zero, zero]],
    1: [[0, finf, pet, extdp, extwc, zero, zero, zero]],
}

# -


# +
def build_model(sim_name):
    print(f"Building models for {sim_name}")

    # build MODFLOW 6 files
    sim = flopy.mf6.MFSimulation(
        sim_name=sim_name, version="mf6", exe_name="mf6", sim_ws=sim_ws
    )

    print(sim.sim_path)

    # create tdis package
    tdis_rc = []
    for i in range(nper):
        tdis_rc.append((perlen[i], nstp[i], tsmult[i]))

    flopy.mf6.ModflowTdis(sim, time_units=time_units, nper=nper, perioddata=tdis_rc)

    newtonoptions = ["NEWTON", "UNDER_RELAXATION"]
    # create gwf model
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        newtonoptions=newtonoptions,
        save_flows=True,
    )

    # create iterative model solution and register the gwf model with it
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        complexity="MODERATE",
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

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain,
        filename=f"{gwfname}.dis",
    )

    # initial conditions
    flopy.mf6.ModflowGwfic(
        gwf,
        strt=strt,
        filename=f"{gwfname}.ic",
    )

    # node property flow
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        icelltype=1,
        k=100.0,
        k33=10,
        filename=f"{gwfname}.npf",
    )

    # aquifer storage
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=1,
        ss=1e-5,
        sy=prsity,
        steady_state=steady,
        transient=transient,
        filename=f"{gwfname}.sto",
    )

    # constant head for draining the model
    chdspd = {0: [[(100, 0, 0), chdval, 10.0]]}

    flopy.mf6.ModflowGwfchd(
        gwf,
        auxiliary=["TEMPERATURE"],
        print_flows=True,
        stress_period_data=chdspd,
        pname="CHD-1",
        filename=f"{gwfname}.chd",
    )

    # Unsaturated-zone flow package
    flopy.mf6.ModflowGwfuzf(
        gwf,
        print_flows=True,
        save_flows=True,
        wc_filerecord=gwfname + ".uzfwc.bin",
        simulate_et=False,
        simulate_gwseep=False,
        linear_gwet=False,
        boundnames=False,
        ntrailwaves=15,
        nwavesets=40,
        nuzfcells=len(uzf_pkdat),
        packagedata=uzf_pkdat,
        perioddata=uzf_spd,
        budget_filerecord=f"{gwfname}.uzf.bud",
        pname="UZF-1",
        filename=f"{gwfname}.uzf",
    )

    # output control
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{gwfname}.cbc",
        head_filerecord=f"{gwfname}.hds",
        headprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        filename=f"{gwfname}.oc",
    )

    # ----------------------------------------------------
    # Instantiating MODFLOW 6 GWE model
    # ----------------------------------------------------
    gwe = flopy.mf6.MFModel(
        sim, model_type="gwe6", modelname=gwename, model_nam_file=f"{gwename}.nam"
    )
    gwe.name_file.save_flows = True

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
        filename=f"{gwename}.ims",
    )
    sim.register_ims_package(imsgwe, [gwe.name])

    # Instantiating MODFLOW 6 transport discretization package
    flopy.mf6.ModflowGwedis(
        gwe,
        nogrb=True,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain,
        pname="DIS",
        filename=f"{gwename}.dis",
    )

    # Instantiating MODFLOW 6 transport initial concentrations
    flopy.mf6.ModflowGweic(
        gwe,
        strt=strt_temp,
        pname="IC",
        filename=f"{gwename}.ic",
    )

    # Instantiating MODFLOW 6 transport advection package
    flopy.mf6.ModflowGweadv(gwe, scheme=scheme, pname="ADV", filename=f"{gwename}.adv")

    # Instantiating MODFLOW 6 transport dispersion package
    flopy.mf6.ModflowGwecnd(
        gwe,
        xt3d_off=False,
        alh=dispersivity,
        ath1=dispersivity,
        ktw=ktw * 86400,
        kts=kts * 86400,
        pname="CND",
        filename=f"{gwename}.cnd",
    )

    # Instantiating MODFLOW 6 transport mass storage package
    flopy.mf6.ModflowGweest(
        gwe,
        save_flows=True,
        porosity=prsity,
        heat_capacity_water=cpw,
        density_water=rhow,
        latent_heat_vaporization=lhv,
        heat_capacity_solid=cps,
        density_solid=rhos,
        pname="EST",
        filename=f"{gwename}.est",
    )

    # Instantiating MODFLOW 6 transport source-sink mixing package
    srctype = "AUX"
    auxname = "TEMPERATURE"
    pname = ["CHD-1"]
    # Inpput to SSM is: <pname> <srctype> <auxname>
    sources = [[itm, srctype, auxname] for itm in pname]

    flopy.mf6.ModflowGwessm(
        gwe,
        sources=sources,
        pname="SSM",
        filename=f"{gwename}.ssm",
    )

    # Instantiating MODFLOW 6 energy transport source-sink mixing package
    uzepackagedata = [(iuz, 10.0) for iuz in range(nlay - 1)]
    uzeperioddata = {0: [[0, "INFILTRATION", 10.0]], 1: [[0, "INFILTRATION", 20.0]]}

    flopy.mf6.ModflowGweuze(
        gwe,
        flow_package_name="UZF-1",
        boundnames=False,
        save_flows=True,
        print_input=True,
        print_flows=True,
        print_temperature=True,
        temperature_filerecord=gwename + ".uze.bin",
        budget_filerecord=gwename + ".uze.bud",
        packagedata=uzepackagedata,
        uzeperioddata=uzeperioddata,
        pname="UZE",
        filename=f"{gwename}.uze",
    )

    # Instantiate MODFLOW 6 heat transport output control package
    flopy.mf6.ModflowGweoc(
        gwe,
        pname="OC",
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
    print("Writing input for 1D Danckwerts test model...")
    success = False
    if isinstance(sim, MFSimulation):
        sim.write_simulation(silent=silent)
    else:
        sim.write_input()


# -


# +
@timed
def run_model(sim, silent=True):
    print("running 1D Danckwerts test model...")
    if isinstance(sim, MFSimulation):
        success, buff = sim.run_simulation(silent=silent, report=True)
    else:
        sim.run_model(silent=silent, report=True)
    assert success, pformat(buff)


# -


# +
def plot_sim_vs_analytical_sln(sim):
    print("comparing simulated results to analytical solution...")

    ws = Path(sim.sim_path)  # exdirs[sim.idxsim]

    # check some output...
    wcobj = flopy.utils.HeadFile(ws / f"{gwfname}.uzfwc.bin", text="water-content")
    wc = wcobj.get_alldata()

    # temperature output
    uzeobj = flopy.utils.HeadFile(ws / f"{gwename}.uze.bin", text="TEMPERATURE")
    temps = uzeobj.get_alldata()

    # Cell flows output
    gweflowsobj = flopy.utils.CellBudgetFile(ws / f"{gwename}.cbc")

    # Binary grid file needed for post-processing
    grb_file = ws / f"{gwfname}.dis.grb"

    # UZE flows
    uzeflowsobj = flopy.utils.CellBudgetFile(ws / f"{gwename}.uze.bud")
    flowsadv = uzeflowsobj.get_data(text="FLOW-JA-FACE")

    t = np.linspace(0.0, 100.0, 101)
    z = np.linspace(0.0, 9.9, 99)

    Kts = kts * 86400
    Ktw = 0.0

    steady_wc = wc[1, 0, 0, 0]
    Sw = steady_wc / prsity

    rhoCp_bulk = Sw * prsity * rhowCpw + (1 - prsity) * rhosCps
    Kt_bulk = Sw * prsity * Ktw + (1 - prsity) * Kts
    v = rhowCpw / rhoCp_bulk * q
    D = Kt_bulk / rhoCp_bulk

    qt0 = q * t0
    qtinfil = q * tinfil

    # for converting from J/day to W
    unitadj = 1 / 86400

    conv10 = []
    cond10 = []
    conv50 = []
    cond50 = []
    conv100 = []
    cond100 = []
    analytical_sln = np.zeros((len(t), len(z)))
    simulated_sln = np.zeros((len(t), len(z)))
    for i, tm in enumerate(t):
        if i == 0:
            gweflowjaface = gweflowsobj.get_data(text="FLOW-JA-FACE", kstpkper=(0, 0))
        else:
            gweflowjaface = gweflowsobj.get_data(
                text="FLOW-JA-FACE", kstpkper=(i - 1, 1)
            )
        flowscond = flopy.mf6.utils.postprocessing.get_structured_faceflows(
            gweflowjaface[0][0], grb_file=grb_file
        )
        for j, depth in enumerate(z):
            fluxa = flux_analyt(tm, depth, qt0, qtinfil, v, D)
            analytical_sln[i, j] = fluxa * rhowCpw * unitadj

            (uze1, uze2, floadv) = flowsadv[i][2 * j + 1]
            floadv /= rhowCpw
            flocond = flowscond[2][j][0][0]
            flocond /= rhowCpw

            flo = floadv * rhowCpw * unitadj + flocond * rhowCpw * unitadj
            if i == 10:
                conv10.append(floadv * rhowCpw * unitadj)
                cond10.append(flocond * rhowCpw * unitadj)
            elif i == 50:
                conv50.append(floadv * rhowCpw * unitadj)
                cond50.append(flocond * rhowCpw * unitadj)
            elif i == 100:
                conv100.append(floadv * rhowCpw * unitadj)
                cond100.append(flocond * rhowCpw * unitadj)

            flux = flo / area
            simulated_sln[i, j] = flux

    # A summary figure without parsing advection and conduction
    fig, ax = plt.subplots(ncols=1, figsize=(3, 6))
    ax.plot(analytical_sln[10], z, "-", color="red", label="Analytical")
    ax.plot(simulated_sln[10], z, "-.", color="blue", label="MODFLOW 6")
    # 50th transient stress period
    ax.plot(analytical_sln[50], z, "-", color="red")
    ax.plot(simulated_sln[50], z, "-.", color="blue")
    # last stress period
    ax.plot(analytical_sln[100], z, "-", color="red")
    ax.plot(simulated_sln[100], z, "-.", color="blue")
    # add labels
    ax.text(5.2, 1.95, "10 days", fontsize=10)
    ax.text(6.5, 3.30, "50 days", fontsize=10)
    ax.text(7.25, 4.70, "100 days", fontsize=10)

    plt.gca().invert_yaxis()
    ax.set_xlabel(r"Energy Flux, $\dfrac{Watts}{m^2}$")
    ax.set_ylabel("Depth, $m$")
    plt.minorticks_on()
    plt.axhline(y=0.0)
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()

    # show and/or save figure
    if plot_show:
        plt.show()
    if plot_save:
        fpth = figs_path / f"ex-{gwename}-01.png"
        fig.savefig(fpth, dpi=600)

    # Generate plots corresponding to three different times
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(7.0, 5))

    # 10 days
    # -------
    polys1 = ax1.stackplot(
        z,
        conv10,
        cond10,
        labels=["Convection", "Conduction"],
        colors=["lightseagreen", "lightgreen"],
    )
    for poly in polys1:
        for path in poly.get_paths():
            path.vertices = path.vertices[:, ::-1]
    ax1.set_xlim(0.095 * rhowCpw * unitadj, 0.205 * rhowCpw * unitadj)
    ax1.set_ylim([9.9, 0.0])
    ax1.plot(simulated_sln[10], z, "-", color="blue", linewidth=2)
    ax1.plot(analytical_sln[10], z, "-.", color="red")
    ax1.text(4.9, 0.4, "10 Days")

    legend_elements = [
        Line2D(
            [0],
            [0],
            linestyle="-",
            color="blue",
            lw=2,
            label="MODFLOW 6\nTotal Heat Flux",
        ),
        Line2D([0], [0], linestyle="-.", color="red", lw=1, label="Analytical"),
        Patch(facecolor="lightseagreen", edgecolor="lightseagreen", label="Convection"),
        Patch(facecolor="lightgreen", edgecolor="lightgreen", label="Conduction"),
    ]

    ax1.legend(handles=legend_elements, loc="lower right", frameon=False)
    ax1.set_xlabel(r"Energy Flux, $\dfrac{Watts}{m^2}$")
    ax1.set_ylabel("Depth, m")

    # 50 days
    # -------
    polys2 = ax2.stackplot(
        z,
        conv50,
        cond50,
        labels=["Convection", "Conduction"],
        colors=["lightseagreen", "lightgreen"],
    )
    for poly in polys2:
        for path in poly.get_paths():
            path.vertices = path.vertices[:, ::-1]
    ax2.set_xlim(0.095 * rhowCpw * unitadj, 0.205 * rhowCpw * unitadj)
    ax2.set_ylim([9.9, 0.0])  # or xlims[::-1]
    ax2.plot(simulated_sln[50], z, "-", color="blue", linewidth=2)
    ax2.plot(analytical_sln[50], z, "-.", color="red")
    ax2.set_xlabel(r"Energy Flux, $\dfrac{Watts}{m^2}$")
    ax2.text(4.9, 0.4, "50 Days")

    # 100 days
    # -------
    polys3 = ax3.stackplot(
        z,
        conv100,
        cond100,
        labels=["Convection", "Conduction"],
        colors=["lightseagreen", "lightgreen"],
    )
    for poly in polys3:
        for path in poly.get_paths():
            path.vertices = path.vertices[:, ::-1]
    ax3.set_xlim(0.095 * rhowCpw * unitadj, 0.205 * rhowCpw * unitadj)
    ax3.set_ylim([9.9, 0.0])  # or xlims[::-1]
    ax3.plot(simulated_sln[100], z, "-", color="blue", linewidth=2)
    ax3.plot(analytical_sln[100], z, "-.", color="red")
    ax3.set_xlabel(r"Energy Flux, $\dfrac{Watts}{m^2}$")
    ax3.text(4.9, 0.4, "100 Days")

    plt.tight_layout()

    # show and/or save figure
    if plot_show:
        plt.show()
    if plot_save:
        fpth = figs_path / f"ex-{gwename}-02.png"
        fig.savefig(fpth, dpi=600)


# -

# +
# ### Running the example
#
# Define a function to run the example scenarios and plot results.


def scenario(silent=True):
    sim = build_model(sim_name)
    if isinstance(sim, MFSimulation) and write:
        write_model(sim, silent=silent)
    if run:
        run_model(sim, silent=silent)
    if plot:
        plot_sim_vs_analytical_sln(sim)


# -

# +
# Initiate model construction, write, run, and plot sequence
scenario(silent=True)
# -
