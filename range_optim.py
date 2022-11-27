# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:51:28 2022

@author: usern
"""
"""
Variables
"""

import numpy as np
import matplotlib.pyplot as plt
from sig import sig
from variables import *
from performance_func import *


rho = rho_cruise
W = W_max
step = 1
C_L = 0.872
distance = 4800*1852 #m
fuel_allow = 487919.0187*0.453592 #kg






x=0
def Range_Iter(W_to, climb_fuel, fuel_allow, V, step, distance_req, delta_optim = 0):
    """
    delta_optim : manual optimisation value
    """
    W_cruise = W_to - climb_fuel
    iter1 = Range_Num(W_cruise, rho_cruise, V, step, distance_req) 

    fuel_allow_new = iter1[0] + iter1[3] + climb_fuel - delta_optim

    W_new = W_to - fuel_allow + fuel_allow_new

    W_new_cruise = W_new - climb_fuel

    iter2 = Range_Num(W_new_cruise, rho_cruise, V, step, distance_req)

    W_loss = iter2[0]
    reserves = iter2[3]
    dist_travel = iter2[2]
    mission_time = iter2[1]
    W_array = iter2[4]
    C_L_array = iter2[5]
    D_array = iter2[6]
    fuelLossArray = iter2[7] + fuel_allow_new
    #fuelAllow_new = reserves + W_loss 

    time_array = np.arange(0, mission_time+step, step) / 3600 # hours


    C_L_avg = sig(np.average(C_L_array), sf)

    ## imperial conversion ##
    if metric:
        W_new = sig(W_new*lb, sf)
        fuel_allow_new = sig(fuel_allow_new*lb, sf)
        reserves = sig(reserves*lb, sf)
        W_array = W_array*lb
        fuelLossArray = fuelLossArray*lb
        C_L_array = C_L_array*kts
        D_array = D_array*lbf
        D_avg = sig(np.average(D_array), sf)
        W_avg = sig(np.average(W_array), sf)
        W_loss = sig(W_loss*lb, sf)
        dist_travel = sig(dist_travel*nmi, sf)

    label = "Mission "+str(x)
    if x == 0:
        label = "Critical Mission"
    plt.figure(0)
    plt.plot(time_array, C_L_array,label=label)
    plt.xlabel('Time (h)')
    plt.ylabel('Coefficient of Lift (-)')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, time_array[-1]+0.5)

    plt.figure(10)
    plt.plot(time_array, D_array, label=label)
    plt.xlabel('Time (h)')
    plt.ylabel('Cruise Thrust/Drag (lbf)')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, time_array[-1]+0.5)

    plt.figure(20+x)
    plt.grid()
    if x == 0:
        mission_title = 'Critical Mission'
    else:
        mission_title = 'Mission '+str(x)
    plt.title(mission_title+', $W_{to}$ = '+str(W_new))
    plt.plot(time_array, W_array, label='Total mass')
    plt.plot(time_array, fuelLossArray, label='Fuel mass')
    plt.plot(time_array, reserves*np.ones_like(time_array), label='reserve fuel', linestyle='--')
    plt.xlabel('Time (h)')
    plt.ylabel('Aircraft mass (lb)')
    plt.legend()
    plt.xlim(0, time_array[-1]+0.5)

    return W_new, fuel_allow_new, reserves, dist_travel, \
        C_L_avg, D_avg, W_avg, C_L_array, D_array, W_array, mission_time 


C_L = 0.5



climb_fuel = 14422 #iterative - lb
other_fuel_estimate = 3000 #takeoff + landing + taxi

W_cruise = W_max - climb_fuel
# loads for each mission




crit_mission_dist = 4800*1/nmi


V_cruise_min = 370 * (1/kts)




W_avg_prelim = 652456*kg # preliminary value based on estimates

V_min_drag = Range_Parameters(W_avg_prelim)[0]*g

V_cruise = V_min_drag

if V_min_drag > V_max:
    V_cruise = V_max
    
dynamic_pressure = 1/2*rho_cruise*V_cruise**2


W_max = 1032500*kg #kg # - original max-takeoff weight
fuel_allow = W_max  - W_empty - W_max_payload - W_crew
crit = Range_Iter(W_max, climb_fuel, fuel_allow, V_cruise, step, crit_mission_dist, 9000)

W_new = crit[0] 
fuel_allow_new = crit[1]
reserves = crit[2]
dist_travel = crit[3]
W_crit_array = crit[-1]
W_new_cruise = W_new - climb_fuel
W_crit_loss = W_new - crit[-2][-1]
V_avg_crit = crit[4]
D_avg_crit = crit[5]
mission_time_crit = crit[-1]
reserves_actual = fuel_allow_new - W_crit_loss - climb_fuel - other_fuel_estimate

reserves_diff = reserves_actual - reserves

if reserves_diff < 0:
    print('u suck')
#### weight arrays

W_payload_array = np.array([180740, 177109.5, 0, 0])*kg
W_passenger_array = np.array([0, 0, 51200, 0])*kg

distance_req_array = np.array([4800, 4400, 3800, 7000]) * 1 / nmi #meters

loiter_array = np.array([0, 20, 20, 15])*60 # seconds

W_to_array_nofuel = W_empty + W_crew + W_payload_array + W_passenger_array

W_to_array = W_to_array_nofuel + fuel_allow_new*kg


W_to_array_new = np.zeros_like(W_to_array)
dist_array = np.zeros_like(W_to_array)
W_loss_array = np.zeros_like(dist_array)
reserves_array = np.zeros_like(dist_array)
fuel_allow_new_array = np.zeros_like(dist_array)
V_avg_array = np.zeros_like(dist_array)
D_avg_array = np.zeros_like(dist_array)
mission_time_array = np.zeros_like(dist_array)

delta_optim = 16000
for i in range(1, len(W_payload_array)):
    x+=1
    if x == 2:
        delta_optim = 21000
        rho_cruise = 0.49307
    if x == 3:
        delta_optim = 4000
        rho_cruise = 0.45973095 # return rho to old value
    
    mission = Range_Iter(W_to_array[i], climb_fuel, fuel_allow_new, \
                                   V_cruise, step, distance_req_array[i],delta_optim)
    W_to_array_new[i] = mission[0]
    dist_array[i] = mission[3]
    W_loss_array[i] = W_to_array_new[i] - climb_fuel - mission[-2][-1]
    reserves_array[i] = mission[2]
    fuel_allow_new_array[i] = mission[1]
    V_avg_array[i] = mission[4]
    D_avg_array[i] = mission[5]
    mission_time_array[i] = mission[-1]
 
mission_time_array[0] = mission_time_crit + loiter_array[0]
mission_time_array[1] = (mission_time_array[1]*2 + (50*60))
mission_time_array[2] = mission_time_array[2] + loiter_array[2]
mission_time_array[3] = mission_time_array[3] + loiter_array[3]

reserves_array[0] = reserves
W_to_array_new[0] = W_new
W_loss_array[0] = W_crit_loss
reserves_actual_array = fuel_allow_new_array - W_loss_array - other_fuel_estimate
reserves_actual_array[0] = reserves_actual
reserves_diff_array = reserves_actual_array - reserves_array

W_end_cruise = W_to_array_new - climb_fuel - W_loss_array - other_fuel_estimate/2

### loiter endurance

V_min_drag = np.sqrt(2*W_end_cruise*kg / (rho_sl*S) * np.sqrt(K / C_D0))

V_stall_loiter = stall_vel(W_end_cruise*kg, S, rho_sl, 1.5)

if V_min_drag.any() < V_stall_loiter.any():
    V_min_drag = V_stall_loiter

D_min_drag = rho_sl*S*C_D0*V_min_drag**2

loiter_fuel = loiter_array*D_min_drag*C


#W_loss_array[1] = W_loss_array[1]*2
#avg_fuel_array = W_loss_array / mission_time_array

D_avg_array[0] = D_avg_crit
