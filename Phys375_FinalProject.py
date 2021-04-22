# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:39:44 2021

@author: James
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from scipy.interpolate import interp1d



# STAR CONSTANTS
k = 1.381*10**-23 # Boltzmann
h = 6.626*10**-34 # Planck
G = 6.674*10**-11 # Newton Gravity
hbar = h/(2*np.pi) # Reduced Planck
me = 9.109*10**-31 # Electron Mass
mp = 1.673*10**-27 # Proton Mass
msun = 1.989*10**30 # Sun Mass
lsun = 3.828*10**26 # Sun Luminosity
c = 2.998*10**8 # Speed of Light
sigma = 5.670*10**-8 # Stefan Boltzmann
a = 4*sigma/c
gamma = 5/3 # Adiabatic index

X = 0.73 # Mass Fraction of Hydrogens
XCNO = 0.03*X # Mass Fraction of CNO
Y = 0.25 # Mass Fraction of Helium
Z = 1-X-Y # Mass Fraction of Other Metals
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



def Star_Trial(omega, Tc, pc_test, first_step):
    '''
    runs through a trial with the initial conditions given

    Parameters
    ----------
    omega : FLOAT
        The angular velocity.
    Tc : FLOAT
        The central temperature (main sequence parameter).
    pc_test : FLOAT
        The test value of central density.
    first_step: FLOAT
        The first step size taken (varies if needed later with RK45)

    Returns
    -------
    trial_values : TUPLE
        Contains the trial error value along with the surface quantities,
        and the list of all the preceeding values.

    '''
    # Sets all initial conditions
    p = pc_test
    T = Tc
    w = omega
    M = 4/3*np.pi*(first_step**3)*p
    E = Energy_Generation_Rate(p, T)
    L = M*E
    K = Opacity(p, T)
    P = Pressure(p, T)
    dPdT = Pressure_Temperature_Derivative(p, T)
    dPdp = Pressure_Density_Derivative(p, T)
    dTdr = Temperature_Gradient(p, first_step, T, L, P, K, M, w)
    dpdr = Density_Gradient(p, first_step, M, dPdT, dTdr, dPdp, w)
    t = Optical_Depth_Gradient(K, p)*first_step
    
    # Initiates the Runge Kutta class object
    RK_obj = RK45(fun=All_Gradients, t0=first_step, y0=[p, T, M, L, t, w],
                  t_bound=1e20, first_step=first_step, rtol=1e-6)

    # Initiates the list of values
    r_vals = [first_step]
    p_vals = [p]
    dpdr_vals = [dpdr]
    T_vals = [T]
    M_vals = [M]
    L_vals = [L]
    t_vals = [t]

    while M < 1000*msun: # sets limit on mass for eroneous solutions
    
        RK_obj.step()
        
        # Updates variables after next step
        r = RK_obj.t
        p = RK_obj.y[0]
        T = RK_obj.y[1]
        M = RK_obj.y[2]
        L = RK_obj.y[3]
        t = RK_obj.y[4]
        
        dPdT = Pressure_Temperature_Derivative(p, T)
        dPdp = Pressure_Density_Derivative(p, T)
        P = Pressure(p, T)
        K = Opacity(p, T)
        dTdr = Temperature_Gradient(p, r, T, L, P, K, M, w)
        dpdr = Density_Gradient(p, r, M, dPdT, dTdr, dPdp, w)
        
        # Calculates opacity proxy to see if small enough yet
        if K*(p**2)/abs(dpdr) < 0.0001 or (t == t_vals[-1]):
            t_inf = t
            break
    
        # Adds new values to list if not breaked yet
        r_vals.append(r)
        p_vals.append(p)
        T_vals.append(T)
        M_vals.append(M)
        L_vals.append(L)
        t_vals.append(t)
    
    
    # Depending on whether mass didn't exceed limit (non-eroneous solution)
    # it will now interpolate to find radius and return info
    if M < 1000*msun:
        delta_t = [(t_inf - t_val) for t_val in t_vals]
        r_interp = interp1d(delta_t, r_vals, kind='linear')
        p_interp = interp1d(delta_t, p_vals, kind='linear')
        T_interp = interp1d(delta_t, T_vals, kind='linear')
        M_interp = interp1d(delta_t, M_vals, kind='linear')
        L_interp = interp1d(delta_t, L_vals, kind='linear')
        t_interp = interp1d(delta_t, t_vals, kind='linear')
    
    else:
        delta_t = [(t_vals[-1] - t_val) for t_val in t_vals]
        r_interp = interp1d(delta_t, r_vals, kind='linear')
        p_interp = interp1d(delta_t, p_vals, kind='linear')
        T_interp = interp1d(delta_t, T_vals, kind='linear')
        M_interp = interp1d(delta_t, M_vals, kind='linear')
        L_interp = interp1d(delta_t, L_vals, kind='linear')
        t_interp = interp1d(delta_t, t_vals, kind='linear')
        
    r_surf = r_interp(2/3)
    p_surf = p_interp(2/3)
    T_surf = T_interp(2/3)
    M_surf = M_interp(2/3)
    L_surf = L_interp(2/3)
    t_surf = t_interp(2/3)
        
    # Determines which index to cover up until to obtain all the values
    for ind in range(len(r_vals)-1, 0, -1):
        if r_vals[ind] < r_surf:
            up_to_ind = len(r_vals)
            break
    
    # Selects the list up until the index at which no more values included
    all_r_vals = r_vals[0:up_to_ind]
    all_p_vals = p_vals[0:up_to_ind]
    all_T_vals = T_vals[0:up_to_ind]
    all_M_vals = M_vals[0:up_to_ind]
    all_L_vals = L_vals[0:up_to_ind]
    all_t_vals = t_vals[0:up_to_ind]
    
    # Computes the trial solution error between theoretical L and computed L
    theoretical_L = 4*np.pi*sigma*(r_surf**2)*(T_surf**4)
    norm_factor = np.sqrt(theoretical_L*L_surf)
    trial_error = (L_surf - theoretical_L)/norm_factor
    
    # Returns all the relevant values
    trial_vals = (trial_error, r_surf, p_surf, T_surf, M_surf, L_surf, 
                  t_surf, all_r_vals, all_p_vals, all_T_vals, all_M_vals, 
                  all_L_vals, all_t_vals)
    return trial_vals



# Inititates plot for the main sequence
fig, ax = plt.subplots(dpi=300, figsize=(5, 5.5))

err_vals = []
r_surf_vals = []
T_surf_vals = []
M_surf_vals = []
L_surf_vals = []
t_surf_vals = []

Temps = np.linspace(9.2e5, 3.3e7, 7)
Omegas = [0]

# Iterates through different central temperature/omega combinations and runs
# the stellar structure on each
for w in Omegas:
    for T in Temps: 
        pc_low = 0.3*1000
        pc_up = 500*1000
        while pc_up-pc_low > 1e-10:           
            pc_guess = (pc_low + pc_up)/2
            trial = Star_Trial(w, T, pc_guess, 1)
            if trial[0] < 0:
                pc_low = pc_guess
            else:
                pc_up = pc_guess
                
        trial = Star_Trial(w, T, pc_up, 1)
        err = float(trial[0])
        r_surf = float(trial[1])
        T_surf = float(trial[3])
        M_surf = float(trial[4])
        L_surf = float(trial[5])
        t_surf = float(trial[6])
        
        # Computes temp by manually setting equation to 0, in case the value
        # of central density converges before the error converges
        T_surf = (L_surf/(4*np.pi*sigma*r_surf**2))**(1/4)
            
        err_vals.append(err)
        r_surf_vals.append(r_surf)
        T_surf_vals.append(T_surf)
        M_surf_vals.append(M_surf)
        L_surf_vals.append(L_surf)
        t_surf_vals.append(t_surf)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    