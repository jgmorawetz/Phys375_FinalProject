# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:39:44 2021

@author: James
"""



import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45



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
X = 0.7 # Mass Fraction of Hydrogen
XCNO = 0.03*X # Mass Fraction of CNO
Z = 0.034112109 # Mass Fraction of Other Metals
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



def Temperature_Gradient(p, r, T, L, P, K, M):
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

    Returns
    -------
    temperature_gradient : FLOAT
        The gradient of temperature.

    '''
    first_term = 3*K*p*L/(16*np.pi*a*c*(T**3)*(r**2))
    second_term = (1 - 1/gamma)*T*G*M*p/(P*r**2)
    return np.max([-first_term, -second_term])



def Density_Gradient(p, r, M, dPdT, dTdr, dPdp):
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

    Returns
    -------
    density_gradient : FLOAT
        The gradient of density.

    '''
    numerator = G*M*p/(r**2) + dPdT*dTdr
    denominator = dPdp
    density_gradient = -numerator/denominator
    return density_gradient



def All_Gradients(r, all_variables):
    '''
    computes a list containing the gradients for the variables of density,
    temperature, enclosed mass, luminosity and optical depth
    
    Parameters
    ----------
    r : FLOAT
        The radius.
    all_variables : LIST of FLOAT
        The list containing density, temperature, enclosed mass,
        luminosity and optical depth.

    Returns
    -------
    gradient_list : LIST of FLOAT
        The list containing the gradient values.

    '''
    p, T, M, L, t = all_variables
    P = Pressure(p, T)
    K = Opacity(p, T)
    E = Energy_Generation_Rate(p, T)
    dPdp = Pressure_Density_Derivative(p, T)
    dPdT = Pressure_Temperature_Derivative(p, T)
    dTdr = Temperature_Gradient(p, r, T, L, P, K, M)
    dpdr = Density_Gradient(p, r, M, dPdT, dTdr, dPdp)
    dMdr = Mass_Gradient(p, r)
    dLdr = Luminosity_Gradient(p, r, E)
    dtdr = Optical_Depth_Gradient(K, p)
    return [dpdr, dTdr, dMdr, dLdr, dtdr]
    
      
 


# Radius step size
dr = 7*10**4

# Initial values
r = dr
p = 5.856*10**4
T = 8.23*10**6
M = 4*np.pi/3*(r**3)*p
E = Energy_Generation_Rate(p, T)
L = 4*np.pi/3*(r**3)*p*E
K = Opacity(p, T)
P = Pressure(p, T)
dMdr = Mass_Gradient(p, r)
dLdr = Luminosity_Gradient(p, r, E)
dTdr = Temperature_Gradient(p, r, T, L, P, K, M)
dpdr = Density_Gradient(p, r, M, Pressure_Temperature_Derivative(p, T),
                        dTdr, Pressure_Density_Derivative(p, T))
dtdr = Optical_Depth_Gradient(K, p)
t = dtdr*dr

# Initiates the lists
N_steps = 10000

r_vals = np.zeros(N_steps); r_vals[0] = r
p_vals = np.zeros(N_steps); p_vals[0] = p
T_vals = np.zeros(N_steps); T_vals[0] = T
M_vals = np.zeros(N_steps); M_vals[0] = M
L_vals = np.zeros(N_steps); L_vals[0] = L
K_vals = np.zeros(N_steps); K_vals[0] = K
E_vals = np.zeros(N_steps); E_vals[0] = E
P_vals = np.zeros(N_steps); P_vals[0] = P
t_vals = np.zeros(N_steps); t_vals[0] = t
dMdr_vals = np.zeros(N_steps); dMdr_vals[0] = dMdr
dLdr_vals = np.zeros(N_steps); dLdr_vals[0] = dLdr
dTdr_vals = np.zeros(N_steps); dTdr_vals[0] = dTdr
dpdr_vals = np.zeros(N_steps); dpdr_vals[0] = dpdr
dtdr_vals = np.zeros(N_steps); dtdr_vals[0] = dtdr

# Initiates the Runge Kutta sequences
RK_obj = RK45(All_Gradients, r, [p, T, M, L, t], max_step=dr, 
              t_bound=10**12)

for i in range(1, N_steps):
    
    RK_obj.step()
    r = RK_obj.t; r_vals[i] = r
    p = RK_obj.y[0]; p_vals[i] = p
    T = RK_obj.y[1]; T_vals[i] = T
    M = RK_obj.y[2]; M_vals[i] = M
    L = RK_obj.y[3]; L_vals[i] = L
    t = RK_obj.y[4]; t_vals[i] = t
    
    K = Opacity(p, T); K_vals[i] = K
    E = Energy_Generation_Rate(p, T); E_vals[i] = E
    P = Pressure(p, T); P_vals[i] = P
    dMdr = Mass_Gradient(p, r); dMdr_vals[i] = dMdr
    dLdr = Luminosity_Gradient(p, r, E); dLdr_vals[i] = dLdr
    dTdr = Temperature_Gradient(p, r, T, L, P, K, M); dTdr_vals[i] = dTdr
    dpdr = Density_Gradient(p, r, M, Pressure_Temperature_Derivative(p, T),
                        dTdr, Pressure_Density_Derivative(p, T)); dpdr_vals[i] = dpdr
    dtdr = Optical_Depth_Gradient(K, p); dtdr_vals[i] = dtdr

Radius = 602334100
fig1, ax1 = plt.subplots(dpi=300)
ax1.plot(r_vals/Radius, p_vals, '-', linewidth=1)
ax1.set_xlabel('Radius (m)')
ax1.set_ylabel('Density ($kg/m^3$)')
ax1.set_xlim(0, 1.01)
fig1.suptitle('Density vs Radius', weight='bold')
ax1.ticklabel_format(axis='both', scilimits=(0,0))
#ax1.set_yscale('log')
fig1.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Density_Radius.png"))

fig2, ax2 = plt.subplots(dpi=300)
ax2.plot(r_vals/Radius, T_vals, '-', linewidth=1)
ax2.set_xlabel('Radius (m)')
ax2.set_ylabel('Temperature (K)')
fig2.suptitle('Temperature vs Radius', weight='bold')
ax2.ticklabel_format(axis='both', scilimits=(0,0))
ax2.set_xlim(0, 1.01)
#ax2.set_yscale('log')
fig2.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Temperature_Radius.png"))

fig3, ax3 = plt.subplots(dpi=300)
ax3.plot(r_vals/Radius, E_vals, '-', linewidth=1)
ax3.set_xlabel('Radius (m)')
ax3.set_ylabel('Energy Generation Rate ($m^2/s^3$)')
ax3.set_xlim(0, 1.01)
fig3.suptitle('Energy Generation Rate vs Radius', weight='bold')
ax3.ticklabel_format(axis='both', scilimits=(0,0))
fig3.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Energy_Generation_Radius.png"))

fig4, ax4 = plt.subplots(dpi=300)
ax4.plot(r_vals/Radius, K_vals, '-', linewidth=1)
ax4.set_xlabel('Radius (m)')
ax4.set_ylabel('Opacity ($m^2/kg$)')
ax4.set_xlim(0, 1.01)
fig4.suptitle('Rosseland Mean Opacity vs Radius', weight='bold')
ax4.ticklabel_format(axis='both', scilimits=(0,0))
#ax4.set_yscale('log')
fig4.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Opacity_Radius.png"))

fig5, ax5 = plt.subplots(dpi=300)
ax5.plot(r_vals/Radius, P_vals, '-', linewidth=1)
ax5.set_xlabel('Radius (m)')
ax5.set_ylabel('Pressure ($kgm^{-1}s^{-2}$)')
ax5.set_xlim(0, 1.01)
fig5.suptitle('Pressure vs Radius', weight='bold')
ax5.ticklabel_format(axis='both', scilimits=(0,0))
fig5.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Pressure_Radius.png"))

fig6, ax6 = plt.subplots(dpi=300)
ax6.plot(r_vals/Radius, M_vals, '-', linewidth=1)
ax6.set_xlabel('Radius (m)')
ax6.set_ylabel('Mass ($kg$)')
ax6.set_xlim(0, 1.01)
fig6.suptitle('Enclosed Mass vs Radius', weight='bold')
ax6.ticklabel_format(axis='both', scilimits=(0,0))
fig6.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Mass_Radius.png"))

fig7, ax7 = plt.subplots(dpi=300)
ax7.plot(r_vals/Radius, L_vals, '-', linewidth=1)
ax7.set_xlabel('Radius (m)')
ax7.set_ylabel('Luminosity ($kgm^2s^{-3}$)')
ax7.set_xlim(0, 1.01)
fig7.suptitle('Luminosity vs Radius', weight='bold')
ax7.ticklabel_format(axis='both', scilimits=(0,0))
fig7.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Luminosity_Radius.png"))

fig8, ax8 = plt.subplots(dpi=300)
ax8.plot(r_vals/Radius, t_vals, '-', linewidth=1)
ax8.set_xlabel('Radius (m)')
ax8.set_ylabel('Optical Depth')
ax8.set_xlim(0, 1.01)
fig8.suptitle('Optical Depth vs Radius', weight='bold')
ax8.ticklabel_format(axis='both', scilimits=(0,0))
fig8.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Optical_Depth_Radius.png"))

fig9, ax9 = plt.subplots(dpi=300)
ax9.plot(r_vals/Radius, dMdr_vals, '-', linewidth=1)
ax9.set_xlabel('Radius (m)')
ax9.set_ylabel('Mass Gradient ($kg/m$)')
ax9.set_xlim(0, 1.01)
fig9.suptitle('Mass Gradient vs Radius', weight='bold')
ax9.ticklabel_format(axis='both', scilimits=(0,0))
fig9.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Mass_Gradient_Radius.png"))

fig10, ax10 = plt.subplots(dpi=300)
ax10.plot(r_vals/Radius, dLdr_vals, '-', linewidth=1)
ax10.set_xlabel('Radius (m)')
ax10.set_ylabel('Luminosity Gradient ($kgms^{-3}$)')
ax10.set_xlim(0, 1.01)
fig10.suptitle('Luminosity Gradient vs Radius', weight='bold')
ax10.ticklabel_format(axis='both', scilimits=(0,0))
fig10.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Luminosity_Gradient_Radius.png"))

fig11, ax11 = plt.subplots(dpi=300)
ax11.plot(r_vals/Radius, dTdr_vals, '-', linewidth=1)
ax11.set_xlabel('Radius (m)')
ax11.set_ylabel('Temperature Gradient ($K/m$)')
ax11.set_xlim(0, 1.01)
fig11.suptitle('Temperature Gradient vs Radius', weight='bold')
ax11.ticklabel_format(axis='both', scilimits=(0,0))
fig11.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Temperature_Gradient_Radius.png"))

fig12, ax12 = plt.subplots(dpi=300)
ax12.plot(r_vals/Radius, dpdr_vals, '-', linewidth=1)
ax12.set_xlabel('Radius (m)')
ax12.set_ylabel('Density Gradient ($kg/m^4$)')
ax12.set_xlim(0, 1.01)
fig12.suptitle('Density Gradient vs Radius', weight='bold')
ax12.ticklabel_format(axis='both', scilimits=(0,0))
fig12.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Density_Gradient_Radius.png"))

fig13, ax13 = plt.subplots(dpi=300)
ax13.plot(r_vals/Radius, dtdr_vals, '-', linewidth=1)
ax13.set_xlabel('Radius (m)')
ax13.set_ylabel('Optical Depth Gradient ($m^{-1}$)')
ax13.set_xlim(0, 1.01)
fig13.suptitle('Optical Depth Gradient vs Radius', weight='bold')
ax13.ticklabel_format(axis='both', scilimits=(0,0))
fig13.savefig(os.path.join(os.path.abspath(
        r"C:\Users\James\Documents\GitHub\Phys375_FinalProject\Final_Plots"),
        "Optical_Depth_Gradient_Radius.png"))

    














    
    
    
    
    
    
    
    
    
    
    
    
    
    
    