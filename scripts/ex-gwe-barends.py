# ## Barends Problem
#
# This example is roughly patterned after Barends (2010).
#
#  Barends, F.B.J., 2010. Complete Solution for Transient Heat Transport in
#    Porous Media, Following Lauwerier's Concept. Society of Petroleum '
#    Engineers, Annual Technical Conference and Exhibition, Florence, Italy,
#    19-22 September 2010.
#    https://doi.org/10.2118/134670-MS
#
# Below is a diagram of the model cell numbering (assuming delr = 4.0; values
# for other ncol discretizations will be different).  The model setup is such
# that the lower half of the model domain is representative of a "groundwater
# reservoir" that is flowing in a 1D, left-to-right manner with no flow in the
# overburden.  One of the main purposes of the model is to test heat "bleeding"
# (conduction) from the warmer overburden to the flowing reservoir.
#
#           +-------+-------+-------+     +-------+-------+-------+          \
# Cell IDs  |   0   |   1   |   2   | ... |  247  |  248  |  249  | Layer 0   |
# (0-based) +-------+-------+-------+     +-------+-------+-------+           |
#           |  250  |  251  |  252  | ... |  497  |  498  |  499  | Layer 1   |
#           +-------+-------+-------+     +-------+-------+-------+           | -- "Overburden"
#           |  500  |  501  |  502  | ... |  747  |  748  |  749  | Layer 2   |
#           +-------+-------+-------+     +-------+-------+-------+           .
#               .       .       .             .       .       .               .
#               .       .       .             .       .       .               .
#               .       .       .             .       .       .              /
#           +-------+-------+-------+     +-------+-------+-------+          \
#           | 49,250| 49,251| 49,252|     | 49,497| 49,448| 49,449| Layer 197 |
#           +-------+-------+-------+     +-------+-------+-------+           |
#           | 49,500| 49,501| 49,502| ... | 49,747| 49,748| 49,749| Layer 198 | -- "Groundwater reservoir"
#           +-------+-------+-------+     +-------+-------+-------+           |
#           | 49,750| 49,751| 49,752|     | 49,997| 49,998| 49,999| Layer 199 |
#           +-------+-------+-------+     +-------+-------+-------+          /
#                       ----> gw flow direction ---->
#
# NOTE: In the current example, ncol is specified as 250.  If users prefer,
#       this value may be increased to 500, 1,000, or more for a more refined
#       solution.  Currently, ncol is kept low for faster run times.


# +
from pathlib import Path

import flopy
import git
import matplotlib.pyplot as plt
import numpy as np
from modflow_devtools.misc import get_env, timed

# Example name and base workspace
sim_name = "ex-gwe-barends"
gwf_name = "gwf-" + sim_name.split("-")[-1]
gwename = "gwe-" + sim_name.split("-")[-1]

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
time_units = "seconds"

nrow = 1  # Number of rows in the simulation ($-$)
ncol = 250  # Number of columns in the simulation ($-$)
L = 1000  # Length of simulation in X direction ($m$)
delc = 1.0  # Width along the column ($m$)
nlay_res = 100  # Number of layers in the groundwater reservoir ($-$)
nlay_overburden = 100  # Number of layer representing the overburden ($-$)
nlay = nlay_overburden + nlay_res  # Total number of layers ($-$)
top_res = 100.0  # Elevation of the top of the groundwater reservoir ($m$)
H = 100.0  # Reservoir height ($m$)
thk = 1.0  # Unit thickness "into the page" ($m$)
k11 = 9.80670e-7  # Hydraulic conductivity of the groundwater reservoir($m/d$)
k33 = 1e-21  # Vertical hydraulic conductivity of the entire domain ($m/s$)
prsity = 0.25  # Porosity ($-$)
scheme = "TVD"  # Advection solution scheme ($-$)
ktw = 0.6  # Thermal conductivity of the fluid ($\dfrac{W}{m \cdot ^{\circ}C}$)
kts = 0.06667  # Thermal conductivity of the aquifer matrix (which is also the dry overburden material) ($\dfrac{W}{m \cdot ^{\circ}C}$)
rhow = 1000.0  # Density of water ($\frac{kg}{m^3}$) # 3282.296651
cpw = 5000.0  # Mass-based heat capacity of the fluid ($\dfrac{J}{kg \cdot ^{\circ}C}$)
rhos = 2000.0  # Density of the solid material ($\dfrac{kg}{m^3}$)
cps = 500.0  # Mass-based heat capacity of the solid material ($\dfrac{J}{kg \cdot $^{\circ}C}$)
alpha_l = 0.0  # Longitudinal dispersivity ($m$)
ath1 = 28.4607479  # Transverse mechanical dispersivity ($m$)
ath2 = 28.4607479  # Transverse mechanical dispersivity ($m$)
T0 = 80  # Initial temperature of the active domain ($^{\circ}C$)
T1 = 30  # Injected temperature ($^{\circ}C}$)
q = 1.2649e-8  # Darcy velocity ($m/s$)


# delr is a calculated value based on the number of desired columns
assert L / ncol % 1 == 0, (
    "reconsider specification of NCOL such that length of the simulation divided by NCOL results in an even number value"
)
delr = L / ncol  # Width along the row ($m$)

# Calculated values
Q_well = q * H * thk  # Injection flow rate ($m^3/day$)


# Calculate grid characteristics
top_overburden = top_res * 2
k11 = [k33 for i in range(nlay_overburden)] + [k11 for i in range(nlay_res)]
icelltype = 0
idomain = 1


# The flow simulation just needs 1 steady-state stress period
# Transport simulation is transient, attempting to solve 200 years in 1 stress period with 10 time steps
nper = 1  # Number of simulated stress periods ($-$)
perlen = 86400.0 * 365 * 200  # Period length ($seconds$)
nstp = 10  # Number of time steps per stress period ($-$)
tsmult = 1.62088625512342  # Time step multiplier ($-$)

tdis_rc = [(perlen, nstp, tsmult)]


# Solver parameters
nouter = 50
ninner = 100
hclose = 1e-6
rclose = 1e-5
# -


# +
# A few steps to get data ready for setting up the model grid

# For the vertical discretization, use a geometric multiplier to refine discretization at
# the reservoir-overburden interface
# For 100 layers with the first cell being 0.01 m thick, use: 1.0673002615
# For 50 layers with the first cell being 0.02 m thick, use: 1.14002980807597
dismult = 1.0673002615

botm_seq = [0.01]  # Starting thickness for the geometric expansion of layer thicknesses

for k in np.arange(1, nlay_overburden):
    botm_seq.append(botm_seq[-1] * dismult)

# Reverse the order for the overburden since botm is top -> down
botm_seq_overburden = list(reversed(botm_seq))

# Define botm for overburden
botm_overburden = [top_overburden - botm_seq_overburden[0]]
for thk_interval in botm_seq_overburden[1:]:
    next_elv = botm_overburden[-1] - thk_interval
    botm_overburden.append(next_elv)

botm_overburden = [round(itm, 2) for itm in botm_overburden]

# Continue moving down the profile and determine discretization for the sat zone
if nlay_res == 1:
    botm_res = [0]
else:
    botm_res = [top_res - botm_seq[0]]
    for thk_interval in botm_seq[1:]:
        next_elv = botm_res[-1] - thk_interval
        botm_res.append(next_elv)

    botm_res = [abs(round(itm, 2)) for itm in botm_res]


# Merge the two botm lists for coming up with a single unified botm object
# to pass to the discretization constructor
botm = botm_overburden + botm_res

# Establish a 2D array of the nodal locations
z_tops = [top_overburden] + botm
z_diffs = abs(np.diff(z_tops))
z_mids = z_tops[:-1] - (z_diffs / 2)

# Based on the way the analytical solution works, the interface altitude
# equals 0.  Moreover, the temperature at the interface equals the
# temperature in the 1D groundwater reservoir.  With that in view,
# set the last element to the interface altitude and subtract off the
# elevation of the top of the gw reservoir, which is z-axis point of
# reference
z_mids[-1] = top_res
z_mids = z_mids - 100

x_mids = [j * delr + (delr / 2) for j in np.arange(ncol)]
X, Y = np.meshgrid(x_mids, z_mids)

# -


# +
# Establish flow boundary conditions. To get a uniform flow field, need to
# thickness-weight the WEL-inserted/extracted flows since the layer thicknesses
# are not uniform
wel_spd_left = []
wel_spd_right = []
ctp_left = []
res_thickness = top_res - botm_res[-1]
for ly in np.arange(0, nlay_res):
    # calculate the thickness-weighting factor
    if nlay_res <= 1:
        thick_interval = top_res - botm_res[ly]
    else:
        thick_interval = botm[(100 + ly) - 1] - botm[100 + ly]

    Qcell = Q_well * thick_interval / res_thickness
    id_left = (100 + ly, 0, 0)
    id_right = (100 + ly, 0, ncol - 1)
    # 30.0 is inflow temperature (auxiliary var)
    wel_spd_left.append([id_left, Qcell, T1])
    wel_spd_right.append([id_right, -Qcell])

    ctp_left.append([id_left, T1])

wel_spd_left = {0: wel_spd_left}
wel_spd_right = {0: wel_spd_right}
ctp_spd = {0: ctp_left}

# -


# +
@timed
def build_mf6_flow_model():
    sim_ws = workspace / sim_name / "mf6gwf"

    # Instantiate a MODFLOW 6 simulation
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")

    # Instantiate time discretization package
    flopy.mf6.ModflowTdis(
        sim,
        nper=1,  # just one steady state stress period for gwf
        perioddata=[(1.0, 1, 1.0)],
        time_units=time_units,
    )

    # Instantiate Iterative model solution package
    flopy.mf6.ModflowIms(
        sim,
        linear_acceleration="bicgstab",
        outer_maximum=nouter,
        outer_dvclose=hclose,
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=f"{rclose} strict",
    )

    # Instantiate a groundwater flow model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwf_name, save_flows=True)

    # Instantiate an unstructured discretization package
    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nogrb=True,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        idomain=idomain,
        top=top_overburden,
        botm=botm,
        pname="DIS",
        filename=f"{gwf_name}.dis",
    )

    # Instantiate node-property flow (NPF) package
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_saturation=True,
        save_specific_discharge=True,
        xt3doptions=False,
        icelltype=icelltype,
        k=k11,
        k33=k33,
        pname="NPF-1",
        filename=f"{gwf_name}.npf",
    )

    # Instantiate initial conditions package for the GWF model
    flopy.mf6.ModflowGwfic(
        gwf, strt=top_overburden, pname="IC-1", filename=f"{gwf_name}.ic"
    )

    # Instantiating MODFLOW 6 storage package
    # (steady flow conditions, so no actual storage,
    # using to print values in .lst file)
    flopy.mf6.ModflowGwfsto(
        gwf,
        ss=0,
        sy=0,
        steady_state={0: True},
        pname="STO",
        filename=f"{gwf_name}.sto",
    )

    # Instantiate WEL package for adding water on the left and removing
    # it on the right
    flopy.mf6.ModflowGwfwel(
        gwf,
        auxiliary="TEMPERATURE",
        stress_period_data=wel_spd_left,
        pname="WEL-left",
        filename=f"{gwf_name}.wel-left",
    )
    # Instantiate WEL package for extracting water on the right
    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=wel_spd_right,
        pname="WEL-right",
        filename=f"{gwf_name}.wel-right",
    )

    # Instantiating MODFLOW 6 output control package (flow model)
    head_filerecord = f"{sim_name}.hds"
    budget_filerecord = f"{sim_name}.cbc"
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    return sim


# -


# +
def build_mf6_heat_model():
    print(f"Building mf6gwe model...{sim_name}")
    sim_ws = workspace / sim_name / "mf6gwe"

    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")

    # Instantiating MODFLOW 6 groundwater transport model
    gwe = flopy.mf6.MFModel(
        sim,
        model_type="gwe6",
        modelname=gwename,
        model_nam_file=f"{gwename}.nam",
    )

    # Instantiate Iterative model solution package
    imsgwe = flopy.mf6.ModflowIms(
        sim,
        linear_acceleration="bicgstab",
        complexity="SIMPLE",
        outer_maximum=nouter,
        outer_dvclose=hclose,
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=f"{rclose} strict",
    )
    sim.register_ims_package(imsgwe, [gwe.name])

    # MF6 time discretization differs from corresponding flow simulation
    flopy.mf6.ModflowTdis(
        sim, nper=len(tdis_rc), perioddata=tdis_rc, time_units=time_units
    )

    # Instantiate an unstructured discretization package
    flopy.mf6.ModflowGwedis(
        gwe,
        length_units=length_units,
        nogrb=True,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        idomain=idomain,
        top=top_overburden,
        botm=botm,
        pname="DIS",
        filename=f"{gwename}.dis",
    )

    # Instantiating MODFLOW 6 heat transport initial temperature
    flopy.mf6.ModflowGweic(gwe, strt=T0, pname="IC-gwe", filename=f"{gwename}.ic")

    # Instantiating MODFLOW 6 heat transport advection package
    flopy.mf6.ModflowGweadv(gwe, scheme=scheme, pname="ADV", filename=f"{gwename}.adv")

    # Instantiating MODFLOW 6 heat transport dispersion package
    if ktw != 0:
        flopy.mf6.ModflowGwecnd(
            gwe,
            alh=alpha_l,
            ath1=ath1,
            ath2=ath2,
            ktw=ktw,
            kts=kts,
            pname="CND",
            filename=f"{gwename}.cnd",
        )

    # Instantiating MODFLOW 6 heat transport mass storage package (consider renaming to est)
    flopy.mf6.ModflowGweest(
        gwe,
        porosity=prsity,
        heat_capacity_solid=cps,
        heat_capacity_water=cpw,
        density_solid=rhos,
        density_water=rhow,
        pname="EST",
        filename=f"{gwename}.est",
    )

    flopy.mf6.ModflowGwectp(
        gwe,
        save_flows=True,
        print_flows=True,
        maxbound=1,
        stress_period_data=ctp_spd,
        pname="CTP",
        filename=f"{gwename}.ctp",
    )

    # Instantiating MODFLOW 6 source/sink mixing package for dealing with
    # auxiliary temperature specified in WEL boundary package.
    sourcerecarray = [("WEL-left", "AUX", "TEMPERATURE")]
    flopy.mf6.ModflowGwessm(
        gwe,
        print_flows=True,
        sources=sourcerecarray,
        pname="SSM",
        filename=f"{gwename}.ssm",
    )

    # Instantiating MODFLOW 6 heat transport output control package
    # e.g., day 100 = 100 * 86400 = 8,640,000 = 8.64e6
    flopy.mf6.ModflowGweoc(
        gwe,
        budget_filerecord=f"{gwename}.cbc",
        temperature_filerecord=f"{gwename}.ucn",
        temperatureprintrecord=[("COLUMNS", 10, "WIDTH", 15, "DIGITS", 6, "GENERAL")],
        saverecord={
            0: [
                ("TEMPERATURE", "LAST"),
                ("BUDGET", "LAST"),
            ]
        },
        printrecord={0: [("BUDGET", "LAST")]},
    )

    # Instantiating MODFLOW 6 Flow-Model Interface package
    pd = [
        ("GWFHEAD", "../mf6gwf/" + sim_name + ".hds", None),
        ("GWFBUDGET", "../mf6gwf/" + sim_name + ".cbc", None),
    ]
    flopy.mf6.ModflowGwefmi(gwe, packagedata=pd)

    return sim


# -


# +
def write_mf6_models(sim_gwf, sim_gwe, silent=True):
    # Run the steady-state flow model
    if sim_gwf is not None:
        sim_gwf.write_simulation(silent=silent)

    # Second, run the heat transport model
    if sim_gwe is not None:
        sim_gwe.write_simulation(silent=silent)


def run_model(sim, silent=True):
    # Attempting to run model
    success, buff = sim.run_simulation(silent=silent, report=True)
    if not success:
        print(buff)
    return success


# -


# +
@timed
def plot_thermal_bleeding(sim_gwe):
    gwe = sim_gwe.get_model(gwename)

    # Retrieve simulated temperature field
    temperature = gwe.output.temperature().get_alldata()

    figsize = (6, 4)
    fig, ax = plt.subplots(figsize=figsize)
    mx = flopy.plot.PlotCrossSection(
        ax=ax, modelgrid=gwe.modelgrid, line={"Row": 0}, extent=(0, 100, 70, 125)
    )
    mx.plot_grid()

    # save figure
    if plot_show:
        plt.show()
    if plot_save:
        fpth = figs_path / f"{sim_name}-gridView.png"
        fig.savefig(fpth, dpi=600)

    # Next, plot model output
    fontsize = 8
    fig2, ax2 = plt.subplots(figsize=figsize)
    mx2 = flopy.plot.PlotCrossSection(
        ax=ax2, modelgrid=gwe.modelgrid, line={"Row": 0}, extent=(0, 400, 50, 175)
    )
    ax2.axhline(y=100, color="black", linestyle="-")

    cb2 = mx2.plot_array(temperature[-1, :, 0, :], cmap="jet", vmin=30, vmax=80)
    levels = [35, 40, 45, 50, 55, 60, 65, 70, 75]

    manual_locations2 = [
        (10, 102),  # 35
        (40, 103),  # 40
        (20, 110),  # 45
        (40, 112),  # 50
        (20, 120),  # 55
        (40, 124),  # 60
        (20, 130),  # 65
        (40, 135),  # 70
        (20, 152),  # 75
    ]
    cb_sim2 = mx2.contour_array(
        temperature[-1, :, 0, :],
        levels=levels,
        linewidths=0.5,
        colors=[
            "white",
            "white",
            "white",
            "teal",
            "teal",
            "teal",
            "teal",
            "white",
            "white",
        ],
    )

    labels2 = ax2.clabel(
        cb_sim2,
        cb_sim2.levels,
        inline=True,
        inline_spacing=1.0,
        fontsize=fontsize,
        fmt="%1d",
        colors=[
            "white",
            "white",
            "white",
            "teal",
            "teal",
            "teal",
            "teal",
            "white",
            "white",
        ],
        manual=manual_locations2,
    )

    ax2.tick_params(axis="both", which="major", labelsize=fontsize)

    ax2.text(300, 103, "Overburden", color="white", fontsize=fontsize)
    ax2.text(300, 87, "Groundwater\nReservoir", color="white", fontsize=fontsize)
    ax2.set_xlabel("X coordinate, m", fontsize=fontsize)
    ax2.set_ylabel("Elevation, m", fontsize=fontsize)

    # save figure
    if plot_show:
        plt.show()
    if plot_save:
        fpth = figs_path / f"{sim_name}-200yrs.png"
        fig2.savefig(fpth, dpi=600)

    return


# -


# +
def scenario(idx, silent=False):
    sim_gwf = build_mf6_flow_model()
    sim_gwe = build_mf6_heat_model()

    if write and (sim_gwf is not None and sim_gwe is not None):
        write_mf6_models(sim_gwf, sim_gwe, silent=silent)

    if run:
        success = run_model(sim_gwf, silent=silent)
        if success:
            success = run_model(sim_gwe, silent=silent)

    if plot and success:
        plot_thermal_bleeding(sim_gwe)


# -

# +
scenario(0, silent=False)
# -
