# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 12:22:07 2021

@author: James
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45


grav_const = 6.67*10**-11
adiabatic_const = 5/3
stef_boltz_const = 5.67*10**-8
speed_light = 3.0*10**8
a_const = 4*stef_boltz_const/speed_light
planck = 6.626*10**-34
reduc_planck = planck/(2*np.pi)
mass_elec = 9.11*10**-31
mass_prot = 1.67*10**-27
boltz_const = 1.38*10**-23
X = 0.7
XCNO = 0.03*X
#Z = 0.034042836
Z = 0.0341089883#0.0340464004#0.035
Y = 1-X-Z
mu = (2*X + 0.75*Y + 0.5*Z)**-1


def density_gradient(encl_mass, dens, rad, press_temp_deriv, temp_grad,
                     press_dens_deriv):
    numerator = (grav_const*encl_mass*dens)/(rad**2) + press_temp_deriv*temp_grad
    denominator = press_dens_deriv
    return -numerator/denominator


def temperature_gradient(opac, dens, lum, temp, rad, press, encl_mass):
    first_term = (3*opac*dens*lum)/(16*np.pi*a_const*speed_light*(temp**3)*(rad**2))
    second_term = (1 - 1/adiabatic_const)*temp*grav_const*encl_mass*dens/(press*(rad**2))
    return -min(first_term, second_term)
    
    
def mass_gradient(rad, dens):
    return 4*np.pi*(rad**2)*dens
    

def luminosity_gradient(rad, dens, eps):
    return 4*np.pi*(rad**2)*dens*eps
    
    
def optical_depth_gradient(opac, dens):
    return opac*dens


def pressure(dens, temp):
    first_term = ((3*(np.pi**2))**(2/3))/5*((reduc_planck**2)/mass_elec)*(dens/mass_prot)**(5/3)
    second_term = dens*boltz_const*temp/(mu*mass_prot)
    third_term = 1/3*a_const*(temp**4)
    return first_term + second_term + third_term


def pressure_density_derivative(dens, temp):
    first_term = ((3*(np.pi**2))**(2/3))/3*((reduc_planck**2)/(mass_elec*mass_prot))*(dens/mass_prot)**(2/3)
    second_term = boltz_const*temp/(mu*mass_prot)
    return first_term + second_term


def pressure_temperature_derivative(dens, temp):
    first_term = dens*boltz_const/(mu*mass_prot)
    second_term = 4/3*a_const*(temp**3)
    return first_term + second_term


def epsilon(dens, temp):
    PP_chain = (1.07*10**-7)*(dens/10**5)*(X**2)*(temp/10**6)**4
    CNO_cycle = (8.24*10**-26)*(dens/10**5)*X*XCNO*(temp/10**6)**19.9
    return PP_chain + CNO_cycle


def opacity(dens, temp):
    opacity_es = 0.02*(1+X)
    opacity_ff = (1.0*10**24)*(Z+0.0001)*((dens/10**3)**0.7)*(temp**-3.5)
    opacity_H = (2.5*10**-32)*(Z/0.02)*((dens/10**3)**0.5)*(temp**9)
    max_term = max(opacity_es, opacity_ff)
    return (1/opacity_H + 1/max_term)**-1



# Sets the initial conditions
drad = 10**3
rad0 = 1
dens0 = 58556
temp0 = 8.23544*10**6

opac0 = opacity(dens0, temp0)
eps0 = epsilon(dens0, temp0)
press0 = pressure(dens0, temp0)

encl_mass0 = (4*np.pi/3)*(rad0**3)*dens0
lum0 = (4*np.pi/3)*(rad0**3)*dens0*eps0

opt_depth_grad0 = optical_depth_gradient(opac0, dens0)
opt_depth0 = opt_depth_grad0*rad0

mass_grad0 = mass_gradient(rad0, dens0)
lum_grad0 = luminosity_gradient(rad0, dens0, eps0)
temp_grad0 = temperature_gradient(opac0, dens0, lum0, temp0, rad0,
                                  press0, encl_mass0)
dens_grad0 = density_gradient(encl_mass0, dens0, rad0, 
                              pressure_temperature_derivative(dens0, temp0),
                              temp_grad0,
                              pressure_density_derivative(dens0, temp0))


# Sets initial arrays
N_steps = 700000

rad_vals = np.zeros(N_steps); rad_vals[0] = rad0
dens_vals = np.zeros(N_steps); dens_vals[0] = dens0
temp_vals = np.zeros(N_steps); temp_vals[0] = temp0
opac_vals = np.zeros(N_steps); opac_vals[0] = opac0
eps_vals = np.zeros(N_steps); eps_vals[0] = eps0
press_vals = np.zeros(N_steps); press_vals[0] = press0
encl_mass_vals = np.zeros(N_steps); encl_mass_vals[0] = encl_mass0
lum_vals = np.zeros(N_steps); lum_vals[0] = lum0
opt_depth_grad_vals = np.zeros(N_steps); opt_depth_grad_vals[0] = opt_depth_grad0
opt_depth_vals = np.zeros(N_steps); opt_depth_vals[0] = opt_depth0
mass_grad_vals = np.zeros(N_steps); mass_grad_vals[0] = mass_grad0
lum_grad_vals = np.zeros(N_steps); lum_grad_vals[0] = lum_grad0
temp_grad_vals = np.zeros(N_steps); temp_grad_vals[0] = temp_grad0
dens_grad_vals = np.zeros(N_steps); dens_grad_vals[0] = dens_grad0


# Iterates through enough times, by updating the variables each time
for i in range(1, N_steps):
    rad0 += drad; rad_vals[i] = rad0
    dens0 += dens_grad0*drad; dens_vals[i] = dens0
    temp0 += temp_grad0*drad; temp_vals[i] = temp0
    opac0 = opacity(dens0, temp0); opac_vals[i] = opac0
    eps0 = epsilon(dens0, temp0); eps_vals[i] = eps0
    press0 = pressure(dens0, temp0); press_vals[i] = press0
    encl_mass0 += mass_grad0*drad; encl_mass_vals[i] = encl_mass0
    lum0 += lum_grad0*drad; lum_vals[i] = lum0
    opt_depth_grad0 = optical_depth_gradient(opac0, dens0); opt_depth_grad_vals[i] = opt_depth_grad0
    opt_depth0 += opt_depth_grad0*drad; opt_depth_vals[i] = opt_depth0
    mass_grad0 = mass_gradient(rad0, dens0); mass_grad_vals[i] = mass_grad0
    lum_grad0 = luminosity_gradient(rad0, dens0, eps0); lum_grad_vals[i] = lum_grad0
    temp_grad0 = temperature_gradient(opac0, dens0, lum0, temp0, rad0, press0, encl_mass0); temp_grad_vals[i] = temp_grad0
    dens_grad0 = density_gradient(encl_mass0, dens0, rad0, pressure_temperature_derivative(dens0, temp0),
                                  temp_grad0, pressure_density_derivative(dens0, temp0)); dens_grad_vals[i] = dens_grad0
    
    

# Checks by checking gradient
r_vals = rad_vals
p_vals = dens_vals
T_vals = temp_vals; dTdr_vals_gradient = np.gradient(T_vals, r_vals)
K_vals = opac_vals
E_vals = eps_vals
P_vals = press_vals
M_vals = encl_mass_vals
L_vals = lum_vals
dtdr_vals_computed = opt_depth_grad_vals
t_vals = opt_depth_vals
dmdr_vals_computed = mass_grad_vals
dLdr_vals_computed = lum_grad_vals
dTdr_vals_computed = temp_grad_vals
dpdr_vals_computed = dens_grad_vals


# Goes through and makes new copies of the list
dpdr_vals_from_data = np.zeros(N_steps)
dTdr_vals_from_data = np.zeros(N_steps)
dmdr_vals_from_data = np.zeros(N_steps)
dLdr_vals_from_data = np.zeros(N_steps)
dtdr_vals_from_data = np.zeros(N_steps)

for j in range(N_steps):
    dpdr_vals_from_data[j] = -(grav_const*M_vals[j]*p_vals[j]/r_vals[j]**2 + \
                               pressure_temperature_derivative(p_vals[j], T_vals[j])*dTdr_vals_gradient[j])/pressure_density_derivative(p_vals[j], T_vals[j])
    dTdr_vals_from_data[j] = -min(3*K_vals[j]*p_vals[j]*L_vals[j]/(16*np.pi*a_const*speed_light*T_vals[j]**3*r_vals[j]**2), (1- 3/5)*T_vals[j]*grav_const*M_vals[j]*p_vals[j]/(P_vals[j]*r_vals[j]**2))
    dmdr_vals_from_data[j] = 4*np.pi*r_vals[j]**2*p_vals[j]
    dLdr_vals_from_data[j] = 4*np.pi*r_vals[j]**2*p_vals[j]*E_vals[j]
    dtdr_vals_from_data[j] = K_vals[j]*p_vals[j]
    
    
fig, ax = plt.subplots(dpi=300)
ax.plot(r_vals, T_vals, 'g-')
ax.set_yscale('log')

#ax.set_yscale('log')
#ax.plot(r_vals, dtdr_vals_from_data, 'r-')



















"""

drad = 10**4
rad = drad

dens = 5.856*10**4
temp = 8.23*10**6
encl_mass = (4*np.pi/3)*(rad**3)*dens
eps = epsilon(dens, temp)
lum = (4*np.pi/3)*(rad**3)*dens*eps

opac = opacity(dens, temp)
press = pressure(dens, temp)
mass_grad = mass_gradient(rad, dens)
lum_grad = luminosity_gradient(rad, dens, eps)
temp_grad = temperature_gradient(opac, dens, lum, temp, rad, press, encl_mass)
dens_grad = density_gradient(encl_mass, dens, rad, pressure_temperature_derivative(dens, temp),
                             temp_grad, pressure_density_derivative(dens, temp))
opt_dep_grad = optical_depth_gradient(opac, dens)
opt_depth = opt_dep_grad*drad

rad_vals = [rad]
dens_vals = [dens]
temp_vals = [temp]
encl_mass_vals = [encl_mass]
lum_vals = [lum]
opac_vals = [opac]
eps_vals = [eps]
press_vals = [press]
opt_depth_vals = [opt_depth]

mass_grad_vals = [mass_grad]
lum_grad_vals = [lum_grad]
temp_grad_vals = [temp_grad]
dens_grad_vals = [dens_grad]
opt_dep_grad_vals = [opt_dep_grad]




RK_obj = RK45(all_gradients, rad, [dens, temp, encl_mass, lum, opt_depth], max_step=drad, t_bound=10**12)


N_steps = 10**5
for i in range(N_steps):
    
    RK_obj.step()
    
    rad = RK_obj.t; rad_vals.append(rad)
    dens = RK_obj.y[0]; dens_vals.append(dens)
    temp = RK_obj.y[1]; temp_vals.append(temp)
    encl_mass = RK_obj.y[2]; encl_mass_vals.append(encl_mass)
    lum = RK_obj.y[3]; lum_vals.append(lum)
    opt_depth = RK_obj.y[4]; opt_depth_vals.append(opt_depth)
    
    opac = opacity(dens, temp); opac_vals.append(opac)
    eps = epsilon(dens, temp); eps_vals.append(eps)
    press = pressure(dens, temp); press_vals.append(press)
    mass_grad = mass_gradient(rad, temp); mass_grad_vals.append(mass_grad)
    lum_grad = luminosity_gradient(rad, dens, eps); lum_grad_vals.append(lum_grad)
    temp_grad = temperature_gradient(opac, dens, lum, temp, rad, press, encl_mass); temp_grad_vals.append(temp_grad)
    dens_grad = density_gradient(encl_mass, dens, rad, pressure_temperature_derivative(dens, temp),
                                 temp_grad, pressure_density_derivative(dens, temp)); dens_grad_vals.append(dens_grad)
    opt_dep_grad = optical_depth_gradient(opac, dens); opt_dep_grad_vals.append(opt_dep_grad)
"""

