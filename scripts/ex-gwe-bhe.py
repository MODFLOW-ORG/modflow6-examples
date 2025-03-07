# ## Thermal Loading of Borehole Heat Exchangers
#
# This example simulates transient thermal energy loading of multiple borehole 
# heat exchangers in a uniform flow field and compares the results to an analytical solution.
#

# ### Initial Setup
#
# Import dependencies, define the example name and workspace,
# and read settings from environment variables.
from pathlib import Path

import flopy
import git
import matplotlib.pyplot as plt
import numpy as np
from modflow_devtools.misc import get_env, timed
from scipy.special import roots_legendre

# Example name and base workspace
sim_name = "ex-gwe-bhe"
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


# +
# ### Define analytical solution
# using the POINT2 algorithm from Wexler (1992) (equation 76) with Gauss-Legendre quadrature
# as implemented in AdePy (https://github.com/cneyens/adepy/blob/v0.1.0/adepy/uniform/twoD.py)
# The bhe() function transforms the heat transport parameters to solute transport parameters 
# and wraps the point2() function to allow for superposition of multiple BHE's and transient energy loading. 
#

def integrand_point2(tau, x, y, v, Dx, Dy, xc, yc, lamb):
    return 1 / tau * np.exp(-(v**2 / (4 * Dx) + lamb) * tau - (x - xc)**2 / (4 * Dx * tau) - (y - yc)**2 / (4 * Dy * tau))

def point2(c0, x, y, t, v, n, al, ah, Qa, xc, yc, Dm=0, lamb=0, R=1.0, order=100):
    """Compute the 2D concentration field of a dissolved solute from a continuous point source in an infinite aquifer
    with uniform background flow.

    Source: [wexler_1992]_ - POINT2 algorithm (equation 76).
    Source code is lifted from the AdePy package, v0.1.0: https://github.com/cneyens/adepy/blob/v0.1.0/adepy/uniform/twoD.py
    
    The two-dimensional advection-dispersion equation is solved for concentration at specified `x` and `y` location(s) and
    output time(s) `t`. A point source is continuously injecting a known concentration `c0` at known injection rate `Qa` in the infinite aquifer
    with specified uniform background flow in the x-direction. It is assumed that the injection rate does not significantly alter the flow
    field. The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed in time and space.

    If multiple `x` or `y` values are specified, only one `t` can be supplied, and vice versa.

    A Gauss-Legendre quadrature of order `order` is used to solve the integral. For `x` and `y` values very close to the source location
    (`xc-yc`) the algorithm might have trouble finding a solution since the integral becomes a form of an exponential integral. See [wexler_1992]_.

    Parameters
    ----------
    c0 : float
        Point source concentration [M/L**3]
    x : float or 1D or 2D array of floats
        x-location(s) to compute output at [L].
    y : float or 1D or 2D array of floats
        y-location(s) to compute output at [L].
    t : float or 1D or 2D array of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    n : float
        Aquifer porosity. Should be between 0 and 1 [-].
    al : float
        Longitudinal dispersivity [L].
    ah : float
        Horizontal transverse dispersivity [L].
    Qa : float
        Volumetric injection rate (positive) of the point source per unit aquifer thickness [L**2/T].
    xc : float
        x-coordinate of the point source [L].
    yc : float
        y-coordinate of the point source [L].
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).
    order : integer, optional
        Order of the Gauss-Legendre polynomial used in the integration. Defaults to 100.

    Returns
    -------
    ndarray
        Numpy array with computed concentrations at location(s) `x` and `y` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    Dx = al * v + Dm
    Dy = ah * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    Dx = Dx / R
    Dy = Dy / R
    Qa = Qa / R

    if len(t) > 1 and (len(x) > 1 or len(y) > 1):
        raise ValueError('If multiple values for t are specified, only one x and y value are allowed')

    root, weights = roots_legendre(order)

    def integrate(t, x, y):
        F = integrand_point2(root*(t-0)/2 + (0+t)/2, x, y, v, Dx, Dy, xc, yc, lamb).dot(weights) * (t - 0)/2
        return F
    
    integrate_vec = np.vectorize(integrate)

    term = integrate_vec(t, x, y)
    term0 = Qa / (4 * n * np.pi * np.sqrt(Dx * Dy)) * np.exp(v * (x - xc) / (2 * Dx))
    
    return c0 * term0 * term


def bhe(Finj,
        x,
        y,
        t,
        xc,
        yc,
        v,
        n,
        rho_s,
        c_s,
        k_s,
        rho_w=1000,
        c_w=4180,
        k_w=0.56,
        al=0,
        ah=0,
        T0=0,
        order=100,
        ):
    
    
    Finj = np.atleast_2d(Finj)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)
    xc = np.atleast_1d(xc)
    yc = np.atleast_1d(yc)

    inj_time = Finj[:,0]
    Finj = Finj[:,1:]
    nbeo = Finj.shape[1]

    if not len(xc) == len(yc) == nbeo:
        raise ValueError('xc should yc have the same length, equal to the number of BEO wells.')
    
    # Compute corresponding solute transport parameters
    kd = c_s / (c_w * rho_w)
    k0 = n * k_w + (1 - n) * k_s
    Dm = k0 / (n * rho_w * c_w)
    rho_b = (1 - n) * rho_s
    R = 1 + kd * rho_b / n
      
    # define mass injection rates
    Finj = Finj / (rho_w * c_w) # W/m / (kg/m3 * J/kg/Kelvin) = J/s/m / (kg/m3 * J/kg/Kelvin) = m2/s * kelvin
    Qa = 1.0 # [L**2/T], unity
    
    # function to calculate the temperature changes for all wells at a given time
    def calculate_temp(inj, ti):
        for i in range(len(inj)):
            if i == 0:
                c = point2(c0=inj[i], 
                        x=x, 
                        y=y, 
                        t=ti, 
                        v=v, 
                        n=n, 
                        al=al, 
                        ah=ah, 
                        Qa=Qa, 
                        xc=xc[i], 
                        yc=yc[i],
                        Dm=Dm,
                        R=R,
                        order=order,
                        )
            else:
                c += point2(c0=inj[i], 
                        x=x, 
                        y=y, 
                        t=ti, 
                        v=v, 
                        n=n, 
                        al=al, 
                        ah=ah, 
                        Qa=Qa, 
                        xc=xc[i], 
                        yc=yc[i],
                        Dm=Dm,
                        R=R,
                        order=order
                        )
        
        return c
    
    # calculate
    if len(t) == 1: # snapshot model
        inj_time = inj_time[inj_time <= t] # drop loading times after requested simulation time for speed-up
        if len(inj_time) == 0:
            raise ValueError('No loading times prior to t.')
        
        for ix, tinj in enumerate(inj_time):
            if ix == 0:
                temp = calculate_temp(Finj[ix], t - tinj)
            else:
                temp += calculate_temp(Finj[ix] - Finj[ix - 1], t - tinj)

    elif (len(x) > 1 or len(y) > 1):
        raise ValueError('If multiple values for t are specified, only one x and y value are allowed') # from point2()

    else: # time series at one location
        inj_time = inj_time[inj_time <= np.max(t)] # drop loading times after maximum requested simulation time for speed-up
        if len(inj_time) == 0:
            raise ValueError('No loading times prior to t.')
        
        for ix, tinj in enumerate(inj_time):
            tix = t > tinj
            nt = len(t[tix])
            if ix == 0:
                temp = calculate_temp(Finj[ix], t - tinj)
            elif nt > 0:
                temp[tix] = temp[tix] + calculate_temp(Finj[ix] - Finj[ix - 1], t[tix] - tinj)


    return temp + T0

# ### MODFLOW
# example 

# moderately advective system, with a square model region
center = (40, 40)
Lx = 2 * center[0]
Ly = 2 * center[1]

crds = np.array([[-5.5, -0.5, 4.5, -2.5, 2.5,],
                [2.5, 0.5, -1.5, 4.5, 2.5,]])

delr = 1.0
delc = 1.0

# place the coordinates in the center of the field so they coincide with the 1x1 m cell centriods
xc = crds[0] + center[0] # + delr / 2
yc = crds[1] + center[1] # + delc / 2

# create output mesh for analytical contours
xg, yg = np.meshgrid(np.linspace(0, Lx, 100), np.linspace(0, Ly, 100))

# time-varying energy loads (W/m) (loosely based on the energy demands in Al-Khoury et al, 2021, fig. 8)
loads = -np.vstack([np.repeat(50, len(xc)),    # january-february
                   np.repeat(37.5, len(xc)),  # march-april
                   np.repeat(0.0, len(xc)),  # may-june
                   np.repeat(-50.0, len(xc)), # july-august
                   np.repeat(-10.0, len(xc)), # september-october
                   np.repeat(45, len(xc)),  # november-december
                  ])
nyear = 3 # repeat three years
loads = np.vstack([loads] * nyear)
time = np.linspace(0, nyear * 365 - 60, nyear * 6) * 86400 # start time of injection phase, first one should equal 0.0
Finj_tv = np.column_stack([time, loads])


v = 2 * 6e-7 # 10 cm/d
# Finj = [100] * len(xc) # W / m
n = 0.2
rho_s = 2650
c_s = 900
k_s = 2.5
rho_w = 1000
c_w = 4180
k_w = 0.56
al = 0
ah = 0
T0 = 0 # background temperature

obs = (50 + delr / 2, 40 + delc / 2) # x-y coordinates of observation point
obs_time = np.linspace(0.0, nyear * 365, 100) * 86400 # observation times

# plot energy loading
plt.bar(time/86400, loads[:,1], width = 60, align = 'center', edgecolor='black')
plt.xlabel('Time (d)')
plt.ylabel('Injection rate (W/m)')
plt.grid(linewidth=0.2)


# MODFLOW input
nrow = int(Ly / delc)
ncol = int(Lx / delr)
nlay = 1.0
top = 1.0
botm = 0.0

# boundary conditions
k = 10 / 86400
grad = v * n / k 
hL = 10
hR = hL - (Lx - delr) * grad

nstp = 10 # steps per stress-period
tsmlt = 1.2

flow_ws = './model_tr/flow'
heat_ws = './model_tr/heat'

# flow
sim = flopy.mf6.MFSimulation(sim_name='beo_flow', sim_ws = flow_ws)
tdis = flopy.mf6.ModflowTdis(sim, nper=1,  perioddata=[(1.0, 1, 1.0)])   
ims = flopy.mf6.ModflowIms(sim, complexity='SIMPLE', inner_dvclose=1e-6, rcloserecord=[1e-6, 'STRICT'])

gwf = flopy.mf6.ModflowGwf(sim, modelname='flow', save_flows=True)


dis = flopy.mf6.ModflowGwfdis(gwf, nrow=nrow, ncol=ncol, nlay=nlay, top=top, botm=botm, delr=delr, delc=delc)
npf = flopy.mf6.ModflowGwfnpf(gwf, k=k, icelltype=0, save_specific_discharge=True, save_saturation=True)
sto = flopy.mf6.ModflowGwfsto(gwf, steady_state={0: True})

ic = flopy.mf6.ModflowGwfic(gwf, strt=hL)

chdrec = []
for j in [0, ncol - 1]:
    if j == 0:
        hchd = hL
    else:
        hchd = hR
        
    for i in range(nrow):
        chdrec.append(
            [(0, i, j), hchd, T0]
        )

chd_pname='CHD_0'
chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdrec, auxiliary='TEMPERATURE', pname=chd_pname)
oc = flopy.mf6.ModflowGwfoc(gwf,
                            budget_filerecord='flow.bud',
                            head_filerecord='flow.hds',
                            saverecord=[('BUDGET', 'ALL'), ('HEAD', 'ALL')],
                            printrecord=[('BUDGET', 'ALL')]
                            )

sim.write_simulation()

success, pbuff = sim.run_simulation()
assert success, pbuff

# heat transport
sim = flopy.mf6.MFSimulation(sim_name='beo_heat', sim_ws = heat_ws)

time_mf = np.repeat(np.diff(time)[0], len(time))
tr_periods = [
    (ti, nstp, tsmlt) for ti in time_mf
]
nper = len(tr_periods)
tdis = flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tr_periods)   
ims = flopy.mf6.ModflowIms(sim, complexity='SIMPLE', inner_dvclose=0.001, linear_acceleration='BICGSTAB')

gwe = flopy.mf6.ModflowGwe(sim, modelname='heat', save_flows=True)
dis = flopy.mf6.ModflowGwedis(gwe, nrow=nrow, ncol=ncol, nlay=nlay, top=top, botm=botm, delr=delr, delc=delc)
if v > 0:
    adv = flopy.mf6.ModflowGweadv(gwe, scheme='tvd')
cnd = flopy.mf6.ModflowGwecnd(gwe, alh=al, ath1=ah, ktw=k_w, kts=k_s)
est = flopy.mf6.ModflowGweest(gwe, density_water=rho_w, heat_capacity_water=c_w, porosity=n, heat_capacity_solid=c_s, density_solid=rho_s)

ic = flopy.mf6.ModflowGweic(gwe, strt=T0)

eslrec = {}
for iper in range(nper):
    eslrec_tr = []
    for i in range(len(xc)):
        cid = gwe.modelgrid.intersect(xc[i], yc[i])
        eslrec_tr.append(
            [(0,) + cid, Finj_tv[iper, i + 1]]
        )
    eslrec[iper] = eslrec_tr

esl = flopy.mf6.ModflowGweesl(gwe, stress_period_data=eslrec)
ssm = flopy.mf6.ModflowGwessm(gwe, sources=[chd_pname, 'AUX', 'TEMPERATURE'])

oc = flopy.mf6.ModflowGweoc(gwe, 
                            budget_filerecord='heat.bud',
                            temperature_filerecord='heat.ucn',
                            saverecord=[('BUDGET', 'ALL'), ('TEMPERATURE', 'ALL')],
                            printrecord=[('BUDGET', 'ALL')]
                            )
fmi = flopy.mf6.ModflowGwefmi(gwe,
                              packagedata=[
                                  ('GWFHEAD', '../flow/flow.hds'),
                                  ('GWFBUDGET', '../flow/flow.bud')
                              ]
                              )
sim.write_simulation()

success, pbuff = sim.run_simulation()
assert success, pbuff

# output
output_kper = 8
temp = gwe.output.temperature()
temp_t = temp.get_data(kstpkper=(nstp-1, output_kper))

obs_ij = gwe.modelgrid.intersect(obs[0], obs[1])
obs_temp = temp.get_ts((0,) + obs_ij)

t = temp.get_times()[nstp * output_kper + (nstp - 1)]

# analytical temperature field and time series for time-varying loads
temp_analy = bhe(Finj_tv, xg, yg, t, xc, yc, v, n, rho_s, c_s, k_s, T0=T0)

# analytical temp time series
temp_obs = bhe(Finj_tv, obs[0], obs[1], obs_time, xc, yc, v, n, rho_s, c_s, k_s, T0=T0)

fig, ax = plt.subplots(1, 1, figsize=(10,6))
csa = ax.contour(xg, yg, temp_analy, levels=np.arange(-20, 20, 1) + T0, colors='black', linewidths=0.5, negative_linestyles='solid')
# plt.clabel(csa, fmt='%.2f', fontsize=8)

pmv = flopy.plot.PlotMapView(gwe, ax=ax)
cs = pmv.contour_array(temp_t, levels=np.arange(-20, 20, 1) + T0, colors='red', linewidths=0.5, negative_linestyles='dashed', linestyles='dashed')
plt.clabel(cs, fmt='%.2f', fontsize=8, colors='black')
#pmv.plot_grid(linewidth=0.2)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.scatter(obs[0], obs[1], marker='x', color='green')
ax.scatter(xc, yc, marker='.', color='black')
ax.set_axisbelow(True)
ax.grid()
ax.set_aspect('equal')
plt.title(f't = {t/86400:.2f} d\n {delr} m spacing, v = {v * 86400:.2f} m/d')
ax.set_xlim(30, 65)
ax.set_ylim(30, 55)

h1, _ = csa.legend_elements()
h2, _ = cs.legend_elements()
ax.legend([h1[0], h2[0]], ['Analytical', 'MODFLOW'])

# plt.savefig('./figures/contours_transient_v2.png', dpi=300)

plt.figure(figsize=(10,4))
plt.plot(obs_time / 86400, temp_obs, label = 'Analytical', color='black')
plt.plot(obs_temp[:,0] / 86400, obs_temp[:,1], label = 'MODFLOW', color='red', linestyle='dashed')
plt.xlabel('Time (d)')
plt.ylabel(r'$\Delta$T (°C)')
plt.grid()
plt.legend()
plt.title(f'(x,y) = {obs}\n{delr} m spacing, v = {v * 86400:.2f} m/d')

# plt.savefig('./figures/ts_transient_v2.png', dpi=300)