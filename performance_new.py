#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 10:05:36 2022

@author: jinjo
"""

from variables import *
from performance_func import *
from sig import sig
import csv

run_cruise = True #save sanity



takeoff = Take_Off(T_max, W_to, S_ref, mu, rho_sl, C_L_max, C_D0, K)

takeoff_dist = takeoff[0]*ft
ground_roll = takeoff[1]*ft
ground_rot = takeoff[2]*ft



## CLIMB
#1/2*rho*V**2*S*(C_D0+K*C_L**2) = T_cont
C_L_climb_lim = np.floor( np.sqrt( 1/K * ( (2*T_cont) / (rho_sl*V_max**2*S) - C_D0 ) ) )

C_L_avg = 1.65

V_v = np.zeros(len(W_to))
gamma_climb = np.zeros(len(W_to))
V_climb = np.zeros(len(W_to))

for i in range(len(W_to)):
    if i == 2:
        climb = Climb_Optim(W_to[i], C_L_avg, rho_cruise=rho_28k, cruise_alt=cruise_alt_2)
    else:
        climb = Climb_Optim(W_to[i], C_L_avg, rho_cruise=rho_cruise)
    
    V_v[i] = sig(climb[0], sf)
    gamma_climb[i] = sig(climb[1], sf)
    V_climb[i] = sig(climb[2]*kts, sf)
    
    
# climb fuel consumption

time_climb = cruise_alt / (V_v / (60*ft))

W_loss_climb = time_climb * C * T_cont


W_land = W_to - W_loss_climb - W_loss





distance_req_array = np.array([4800, 4400, 3800, 7000]) * 1 / nmi #meters




if run_cruise:
    
    cruise_drag = np.zeros(len(W_to))
    cruise_time = np.zeros(len(W_to))
    cruise_C_L = np.zeros(len(W_to))
    cruise_loss = np.zeros(len(W_to))
    cruise_reserve = np.zeros(len(W_to))
    
    for i in range(len(W_to)):
        
        rho_temp = rho_cruise
        
        if i == 2:
            rho_temp = rho_28k
        cruise = Range_Num(W_to[i] - W_loss_climb[i], rho_temp, V_max, 0.1, distance_req_array[i])
        
        cruise_drag[i] = sig(np.mean(cruise[-3])*lbf, sf)
        cruise_time[i] = sig(cruise[1] / 3600, sf)
        cruise_C_L[i] = sig(cruise[-1], sf)
        cruise_loss[i] = sig(cruise[0]*lb, sf)
        cruise_reserve[i] = sig(cruise[3]*lb, sf)
        







### loiter
### loiter endurance


C_L_min_drag = np.sqrt(C_D0 / K)

V_min_drag = np.sqrt(2*W_land[1:]*g / (rho_sl*S) * np.sqrt(K / C_D0))

V_stall_loiter = stall_vel(W_land[1:], S, rho_sl, 1.5)

if V_min_drag.any() < V_stall_loiter.any():
    V_min_drag = V_stall_loiter

D_min_drag = Drag(rho_sl, V_min_drag, S, C_D0, K, C_L_min_drag)

loiter_fuel = loiter_array*D_min_drag*C


for i in range(len(loiter_array)):
    V_min_drag[i] = sig(V_min_drag[i]*ft, sf)
    D_min_drag[i] = sig(D_min_drag[i]*lbf, sf)
    loiter_fuel[i] = sig(loiter_fuel[i]*lb, sf)


cruise_time[1] = sig(cruise_time[1]*2, sf)
cruise_loss[1] = sig(cruise_loss[1]*2, sf)






### landing
land = Landing(W_land, S_ref, idle_flow, rho_sl, C_L_max, 0.06)

# (kg/m^3) * (m^2) / (kg*m / s^2)
# (kg / m) / (kg * m / s^2)
# s^2


### making table

for i in range(len(land[0])):
    for j in range(len(land)):
        if j < len(takeoff) - 2:
            land[j][i] = sig(land[j][i]*ft, sf)
        else:
            land[j][i] = sig(land[j][i], sf)    


for i in range(len(takeoff[0])):
    for j in range(len(takeoff)):
        if j < len(takeoff) - 2 or j == 6:
            takeoff[j][i] = sig(takeoff[j][i]*ft, sf)
        else:
            takeoff[j][i] = sig(takeoff[j][i], sf)



### overarching mission

W_cruise_start = W_to - W_loss_climb

for i in range(len(W_cruise_start)):
    W_cruise_start[i] = sig(W_cruise_start[i]*lb, sf)
    W_land[i] = sig(W_land[i]*lb, sf)
    time_climb[i] = sig(time_climb[i] / 60, sf)
    W_loss_climb[i] = sig(W_loss_climb[i]*lb, sf)


"""
for i in range(len(mission_time)):
    mission_time[i] = sig(mission_time[i], sf)
    
mission_array = np.array([W_to*lb, W_land*lb, ])
"""



heading = np.array(['Critical Mission', 'Mission 1', 'Mission 2', 'Mission 3'])

takeoff_head = np.array(['Total Distance (ft)', '$S_G$ (ft)', '$S_R$ (ft)', '$S_TR$ (ft)', \
                         '$h_{TR} (ft)', 'Climb Angle (deg)', 'V_{stall} (ft/s)', \
                             'V_{to} (ft/s)', 'V_{trans} (ft/s)', 'K_A', 'K_T'])
    

    


alt_array = np.array([30000, 30000, 28000, 30000])
climb_array = np.array([alt_array, V_v, V_climb, gamma_climb, time_climb, W_loss_climb])




cruise_array = np.array([W_cruise_start, np.ones_like(W_to)*V_max, distance_req_array *nmi, \
                         cruise_time, cruise_loss, cruise_reserve, \
                             cruise_C_L, cruise_drag])

    
loiter_array = np.array([loiter_array/60, V_min_drag, D_min_drag, loiter_fuel])




np.savetxt('Take-Off.csv', takeoff, delimiter=",")
np.savetxt('Landing.csv', land, delimiter=",")
np.savetxt('Climb.csv', climb_array, delimiter=",")
np.savetxt('Cruise.csv', cruise_array, delimiter=",")
np.savetxt('Loiter.csv', loiter_array, delimiter=".")

