# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 23:54:58 2022

@author: usern

variables
"""
import numpy as np

metric = True



# imperial conversion #
lb = 2.20462 #kg
kts = 1.94384 #m/s
lbf = 1/4.44822 #N
N = 1 / lbf
kg = 1 / lb # lb
nmi = 0.000539957 #m
ft = 3.28084 #m
slugft3 = 0.00194032 #kg/m3


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
S = 4891*1/ft**2 #m
W_max = 1030500 / lb # kg
 
T_max = 388000 * N # take off thrust (N)
T_cont = 332400 * N # continous thrust, newton

rho_cruise = 0.4600 #kg/m^3 -> 30,000 ft
rho_sl = 1.225 #kg/m^3


rho_8000m = 0.5258 #rho at 8000 m
rho_9000m = 0.4671 #rho at 9000 m

cruise_alt_2 = 28000*1/ft

#air density at 28,000 ft
rho_28k = rho_8000m -  ((cruise_alt_2 - 8000) / 1000 * (rho_8000m - rho_9000m) )  


C = (0.478 / 3600) * kg/N   #thrust specific fuel consumption in kg/Ns
idle_flow = 0.301 #kg/sec

S_ref = 5352*1/ft**2 #m

cruise_c = 303 #speed of sound at cruise
 

h_obstacle = 50 / ft #m
runway_dist = 5950 #ft
cruise_alt = 30000 / ft

W_empty = 392435.4223*kg
W_max_payload = 180740*kg
W_crew = 600*kg
    

# take-off weights calculated from numerical methods
W_to = np.array([910000, 823000, 619000, 724000])*kg # weight at take off
W_loss = np.array([283000, 225000, 155000, 294000])*kg

fuel_total = np.array([352000, 253000, 175000, 331000])*kg

fuel_climb_estimate = 14422*kg

    
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
    
    W_empty = 393132 #lb
    W_max_payload = 180740 #lb
    W_crew = 600 #LB
    
    S = 4891
    S_ref = 5352
    
    idle_flow = idle_flow*lb
    
    rho_28k = rho_28k * slugft3
    
    W_to = W_to*lb
    W_loss = W_loss*lb
    fuel_total = fuel_total*lb
    fuel_climb_estimate = fuel_climb_estimate*lb


V_max = mach_max*cruise_c





W_land = W_to - fuel_climb_estimate - W_loss 






    
