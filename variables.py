# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 23:54:58 2022

@author: usern

variables
"""

metric = True



# imperial conversion #
lb = 2.20462 #kg
kts = 1.94384 #m/s
lbf = 1/4.44822 #N
N = 1 / lbf
kg = 1 / lb # lb
nmi = 0.000539957 #m
ft = 3.28084 #m


# unitless
K = 0.055579171
C_L_max = 3.2 #2.7198
mach_max = 0.75 # maximum mach number
mu = 0.05 #rolling resistance
C_D0 = 0.0176 #historical

sf = 3 # number of significant figures to use in final results
step_size = 0.01


### METRIC UNITS ###
g = 9.78 #32.174 #ft/s^2
S = 4891*0.092903 #ft^2
W_max = 1030500 / lb # kg
 
T_max = 388000 * N # take off thrust (N)
T_cont = 332400 * N # continous thrust, newton

rho_cruise = 0.4600 #kg/m^3 -> 30,000 ft
rho_sl = 1.225 #kg/m^3

C = 13.5e-6  #thrust specific fuel consumption in kg/Ns
S_ref = 5352*0.092903

cruise_c = 303 #speed of sound at cruise
 

h_obstacle = 50 / ft #m
runway_dist = 5950 #ft
cruise_alt = 30000 / ft
    
    
if metric == False:
    g = 32.174
    S = S * ft **2 # ft^2
    W_max = W_max * lb # lb
    
    rho_sl = 23.77e-4 #slug/ft^3
    rho_cruise = 8.91e-4 #slug/ft^3 -> 30,000 ft
    cruise_c = cruise_c * ft
    
    h_obstacle = 50 * ft #ft
    T_max = T_max * lbf # lbf
 
    C = 0.478 / 3600 #thrust specific fuel consumption in lb/s
    T_cont = 332400
    runway_dist = runway_dist * 1/ft 
    cruise_alt = cruise_alt / ft


V_max = mach_max*cruise_c








    
