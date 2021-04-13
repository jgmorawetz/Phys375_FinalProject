# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:39:44 2021

@author: James
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import bisect
from scipy.interpolate import interp1d



# STAR CONSTANTS
k = 1.381*10**-23 # Boltzmann
h = 6.626*10**-34 # Planck
G = 6.674*10**-11 # Newton Gravity
hbar = h/(2*np.pi) # Reduced Planck
me = 9.109*10**-31 # Electron Mass
mp = 1.673*10**-27 # Proton Mass
sigma = 5.670*10**-8 # Stefan Boltzmann
c = 3.0*10**8 # Speed of Light
a = 4*sigma/c
gamma = 5/3 # Adiabatic index
dr = 7*10**4 # Radius step size


X = 0.7 # Mass Fraction of Hydrogens
XCNO = 0.03*X # Mass Fraction of CNO
Z = 0.03 # Mass Fraction of Other Metals
Y = 1-X-Z # Mass Fraction of Helium
mu = (2*X + 0.75*Y + 0.5*Z)**-1 # Mean Molecular Mass



def Mass_Gradient(p, r):
    '''
    computes the mass gradient in terms of the density and radius

    Parameters
    ----------
    p : FLOAT
        The density.
    r : FLOAT
        The radius.

    Returns
    -------
    mass_gradient : FLOAT
        The mass gradient.

    '''
    mass_gradient = 4*np.pi*(r**2)*p
    return mass_gradient



def Luminosity_Gradient(p, r, E):
    '''
    computes the gradient of luminosity in terms of the density, radius and
    energy generation rate

    Parameters
    ----------
    p : FLOAT
        The density.
    r : FLOAT
        The radius.
    E : FLOAT
        The energy generation rate.

    Returns
    -------
    luminosity_gradient : FLOAT
        The gradient of luminosity.

    '''
    luminosity_gradient = 4*np.pi*(r**2)*p*E
    return luminosity_gradient
    


def Pressure(p, T):
    '''
    computes the pressure in terms of the density and temperature

    Parameters
    ----------
    p : FLOAT
        The density.
    T : FLOAT
        The temperature.

    Returns
    -------
    pressure : FLOAT
        The pressure.

    '''
    first_term = (3*np.pi**2)**(2/3)/5*(hbar**2)/me*(p/mp)**(5/3)
    second_term = p*k*T/(mu*mp)
    third_term = 1/3*a*(T**4)
    pressure = first_term + second_term + third_term
    return pressure



def Pressure_Density_Derivative(p, T):
    '''
    computes the partial derivative of pressure with respect to density, in
    terms of density and temperature

    Parameters
    ----------
    p : FLOAT
        The density.
    T : FLOAT
        The temperature.

    Returns
    -------
    pressure_density_derivative : FLOAT
        The partial derivative of pressure with respect to density.

    '''
    first_term = (3*np.pi**2)**(2/3)/3*(hbar**2)/(me*mp)*(p/mp)**(2/3)
    second_term = k*T/(mu*mp)
    pressure_density_deriv = first_term + second_term
    return pressure_density_deriv
    


def Pressure_Temperature_Derivative(p, T):
    '''
    computes the partial derivative of pressure with respect to temperature,
    in terms of density and temperature

    Parameters
    ----------
    p : FLOAT
        The density.
    T : FLOAT
        The temperature.

    Returns
    -------
    pressure_temp_deriv : FLOAT
        The partial derivative of pressure with respect to temperature.

    '''
    first_term = p*k/(mu*mp)
    second_term = 4/3*a*(T**3)
    pressure_temp_deriv = first_term + second_term
    return pressure_temp_deriv



def Energy_Generation_Rate(p, T):
    '''
    computes the total energy generation rate, including the PP-chain and 
    the CNO cycle, in terms of the density and temperature

    Parameters
    ----------
    p : FLOAT
        The density.
    T : FLOAT
        The temperature.

    Returns
    -------
    energy_gen_rate : FLOAT
        The total energy generation rate.

    '''
    PP_rate = 1.07*10**-7*(p/10**5)*(X**2)*(T/10**6)**4
    CNO_rate = 8.24*10**-26*(p/10**5)*X*XCNO*(T/10**6)**19.9
    energy_gen_rate = PP_rate + CNO_rate
    return energy_gen_rate



def Opacity(p, T):
    '''
    computes the total Rosseland mean opacity in terms of density
    and temperature

    Parameters
    ----------
    p : FLOAT
        The density.
    T : FLOAT
        The temperature.

    Returns
    -------
    total_opacity : FLOAT
        The Rosseland mean opacity.

    '''
    kes = 0.02*(1 + X)
    kff = 1.0*10**24*(Z + 0.0001)*((p/10**3)**0.7)*(T**-3.5)
    kH = 2.5*10**-32*(Z/0.02)*((p/10**3)**0.5)*(T**9)
    opacity_term = 1/max(kes, kff) + 1/kH
    total_opacity = opacity_term**-1
    return total_opacity
    


def Optical_Depth_Gradient(K, p):
    '''
    computes the gradient of optical depth in terms of Rosseland mean
    opacity and the density

    Parameters
    ----------
    K : FLOAT
        The Rosseland mean opacity.
    p : FLOAT
        The density.

    Returns
    -------
    optical_depth_gradient : FLOAT
        The gradient of optical depth.

    '''
    optical_depth_gradient = K*p
    return optical_depth_gradient



def Temperature_Gradient(p, r, T, L, P, K, M, w):
    '''
    computes the temperature gradient in terms of the density, radius, 
    temperature, luminosity, pressure, opacity, and enclosed mass

    Parameters
    ----------
    p : FLOAT
        The density.
    r : FLOAT
        The radius.
    T : FLOAT
        The temperature.
    L : FLOAT
        The total luminosity.
    P : FLOAT
        The total pressure.
    K : FLOAT
        The opacity.
    M : FLOAT
        The enclosed mass.
    w : FLOAT
        The angular velocity (omega).

    Returns
    -------
    temperature_gradient : FLOAT
        The gradient of temperature.

    '''
    first_term = 3*K*p*L/(16*np.pi*a*c*(T**3)*(r**2))
    geff = G*M/(r**2) - 2/3*(w**2)*r
    second_term = (1 - 1/gamma)*T/P*geff*p
    return -min(first_term, second_term)



def Density_Gradient(p, r, M, dPdT, dTdr, dPdp, w):
    '''
    computes the density gradient in terms of the density, radius, enclosed
    mass, partial derivative of pressure with respect to temperature, the 
    temperature gradient and the partial derivative of pressure with respect
    to density

    Parameters
    ----------
    p : FLOAT
        The density.
    r : FLOAT
        The radius.
    M : FLOAT
        The enclosed mass.
    dPdT : FLOAT
        The partial derivative of pressure with respect to temperature.
    dTdr : FLOAT
        The temperature gradient.
    dPdp : FLOAT
        The partial derivative of pressure with respect to density.
    w : FLOAT
        The angular velocity (omega).

    Returns
    -------
    density_gradient : FLOAT
        The gradient of density.

    '''
    geff = G*M/(r**2) - 2/3*(w**2)*r
    numerator = geff*p + dPdT*dTdr
    denominator = dPdp
    density_gradient = -numerator/denominator
    return density_gradient



def All_Gradients(r, all_variables):
    '''
    computes a list containing the gradients for the variables of density,
    temperature, enclosed mass, luminosity, optical depth, and angular velocity
    
    Parameters
    ----------
    r : FLOAT
        The radius.
    all_variables : LIST of FLOAT
        The list containing density, temperature, enclosed mass,
        luminosity, optical depth and angular velocity.

    Returns
    -------
    gradient_list : LIST of FLOAT
        The list containing the gradient values.

    '''
    p, T, M, L, t, w = all_variables
    P = Pressure(p, T)
    K = Opacity(p, T)
    E = Energy_Generation_Rate(p, T)
    dPdp = Pressure_Density_Derivative(p, T)
    dPdT = Pressure_Temperature_Derivative(p, T)
    dTdr = Temperature_Gradient(p, r, T, L, P, K, M, w)
    dpdr = Density_Gradient(p, r, M, dPdT, dTdr, dPdp, w)
    dMdr = Mass_Gradient(p, r)
    dLdr = Luminosity_Gradient(p, r, E)
    dtdr = Optical_Depth_Gradient(K, p)
    return [dpdr, dTdr, dMdr, dLdr, dtdr, w]



def Opacity_Proxy(i, p_vals, dpdr_vals, T_vals):
    '''
    computes the value of the opacity proxy at that index corresponding to 
    a particular density and temperature

    Parameters
    ----------
    i : INT
        The index within the arrays of interest.
    p_vals : NP.ARRAY
        The full list of density values.
    dpdr_vals : NP.ARRAY
        The full list of density gradient values.
    T_vals : NP.ARRAY
        The full list of temperature values.
    
    Returns
    -------
    proxy_value : FLOAT
        The opacity proxy value.

    '''
    dens = p_vals[i]
    temp = T_vals[i]
    opac = Opacity(dens, temp)
    dens_grad = dpdr_vals[i]
    return opac*(dens**2)/abs(dens_grad)



Tc = 8.23544*10**6#2*10**7#8.23544*10**6
omega = 0

def Trial_Error(pc_test):
    '''
    runs through a trial with the initial conditions given

    Parameters
    ----------
    omega : FLOAT
        The angular velocity.
    Tc : FLOAT
        The central temperature (main sequence parameter).
    pc_test : FLOAT
        The test value of central density (trial parameter).

    Returns
    -------
    error : FLOAT
        The error from the current trial.

    '''
    
    p = pc_test
    T = Tc
    w = omega
    M = 4/3*np.pi*(dr**3)*p
    E = Energy_Generation_Rate(p, T)
    L = M*E
    t = Optical_Depth_Gradient(Opacity(p, T), p)
    
    soln = solve_ivp(fun=All_Gradients, t_span=(dr, 40000*dr),
                     y0=[p, T, M, L, t, w], first_step=dr, max_step=dr)
    
    r_vals = soln.t
    T_vals = soln.y[1]
    L_vals = soln.y[3]
    t_vals = soln.y[4]
    
    tau_inf = t_vals[-1] # MUST CHANGE LATER TO ACTUALLY USE PROXY BUT FINE FOR NOW
    delta_tau = [abs(tau_inf-t_vals[i]) for i in range(len(t_vals))]
    
    interp_func = interp1d(delta_tau, list(range(len(r_vals))), kind='nearest')
    surf_ind = int(interp_func(2/3))
    surf_radius = r_vals[surf_ind]
    surf_luminosity = L_vals[surf_ind]
    surf_temp = T_vals[surf_ind]
    theoretical_luminosity = 4*np.pi*sigma*(surf_radius**2)*(surf_temp**4)
    norm_factor = np.sqrt(theoretical_luminosity*surf_luminosity)
    return (surf_luminosity-theoretical_luminosity)/norm_factor
    

pc_soln = bisect(Trial_Error, 0.3*1000, 500*1000, xtol=0.00000001)


#%%
fig, ax = plt.subplots(dpi=300)
ax.set_xlim(0, 1e9)

for p_test in [61341]:#np.linspace(61340.01632, 61340.016354, 10):
    p = p_test
    T = Tc
    w = omega
    M = 4/3*np.pi*(dr**3)*p
    E = Energy_Generation_Rate(p, T)
    L = M*E
    t = Optical_Depth_Gradient(Opacity(p, T), p)
    
    soln = solve_ivp(fun=All_Gradients, t_span=(dr, 10000*dr),
                     y0=[p, T, M, L, t, w], first_step=dr, max_step=dr)
    
    r_vals = soln.t
    p_vals = soln.y[0]
    dpdr_vals = np.gradient(p_vals, r_vals)
    T_vals = soln.y[1]
    M_vals = soln.y[2]
    L_vals = soln.y[3]
    t_vals = soln.y[4]
    P_vals = [Pressure(p_vals[i], T_vals[i]) for i in range(len(T_vals))]
    logP_vals = list(map(lambda x: np.log10(x), P_vals))
    logT_vals = list(map(lambda x: np.log10(x), T_vals))
    dlogPdlogT_vals = np.gradient(logP_vals, logT_vals)
    tau_inf = t_vals[-1] # MUST CHANGE LATER TO ACTUALLY USE PROXY BUT FINE FOR NOW
    delta_tau = [abs(tau_inf-t_vals[i]) for i in range(len(t_vals))]
    interp_func = interp1d(delta_tau, list(range(len(r_vals))), kind='nearest')
    surf_ind = int(interp_func(2/3))
    surf_radius = r_vals[surf_ind-1]
    surf_luminosity = L_vals[surf_ind-1]
    surf_temp = T_vals[surf_ind-1]
    surf_dens = p_vals[surf_ind-1]
    surf_mass = M_vals[surf_ind-1]
    surf_opt_depth = t_vals[surf_ind-1]
    surf_dlogPdlogT = dlogPdlogT_vals[surf_ind-1]
    

    ax.plot(r_vals, T_vals, '-', label='{}'.format(p_test))

ax.legend()
ax.set_yscale('log')


fig2, ax2 = plt.subplots(dpi=300)
ax2.plot(r_vals, dlogPdlogT_vals)
#ax2.set_yscale('log')
ax2.set_xlim(0,1e9)
ax2.set_ylim(0, 10)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    