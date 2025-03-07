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