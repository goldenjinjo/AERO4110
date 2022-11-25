# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:49:13 2022


AERO4110 Code - Performance and Flight Dynamics

@author: usern
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, N, exp, solve, simplify, sqrt
from sig import sig
from performance_func import *
from variables import *

C_L_min_drag = np.sqrt( C_D0 / K)

V_range, D_range, C_L_range = Range_Parameters(W_max*0.9)

L_range = Lift(rho_cruise, V_range, 2, S)

range_max = Range(L_range, D_range, C, W_max, 0.6*W_max,V=V_range)



V_stall = stall_vel(W_max, S, rho_sl, C_L_max)
V_to = 1.1*V_stall

L_to = Lift(rho_sl, V_to, C_L_max, S) # lift at take-off


TW = T_max/W_max #thrust to weight ratio
b_coeff = -mu +(1/1.1**2)*(mu - TW)
root_factor = np.sqrt((b_coeff)**2 - 4*K*C_D0)

C_L_max = 1/(2*K) * (-b_coeff + root_factor)
C_L_min = 1/(2*K) * (-b_coeff - root_factor)

# Reduce bounds on graph
if C_L_min < 0.5:
    C_L_min = 0.5
    
if C_L_max > 4:
    C_L_max = 4

C_L_array = np.arange(C_L_min, C_L_max-0.05, step_size)



K_A_array = np.zeros_like(C_L_array)
K_T_array = np.zeros_like(C_L_array)
V_to_array = np.zeros_like(C_L_array)
gamma_array = np.zeros_like(C_L_array)
h_TR_array = np.zeros_like(C_L_array)
take_off_array = np.zeros_like(C_L_array)
S_TR_array = np.zeros_like(C_L_array)
S_R_array = np.zeros_like(C_L_array)
S_G_array = np.zeros_like(C_L_array)

for i in range(len(C_L_array)):
        K_A_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[-1]
        K_T_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[-2]
        V_to_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[-4]
        gamma_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[5]
        h_TR_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[4]
        take_off_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[0]
        S_G_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[1]
        S_TR_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[3]
        S_R_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[2]




C_L_takeoff = 3.18
takeoff_index = np.where(C_L_array > C_L_takeoff)[0][0]
design_takeoff_dist = take_off_array[takeoff_index]

min_dist = np.min(take_off_array)
min_dist_index = np.where(take_off_array == min_dist)[0][0]
C_L_min_dist = C_L_array[min_dist_index]
min_h_TR = h_TR_array[min_dist_index]
min_gamma = np.degrees(gamma_array[min_dist_index])

min_clearance = min_h_TR - h_obstacle


if min_dist > runway_dist:
    print('FAILED: Runway not cleared.')
else:
    print('SUCCESS: Runway cleared.')
    min_C_L_for_takeoff = C_L_array[np.where(take_off_array < runway_dist)[0][0]]
    print('C_L for take-off distance of exactly ', runway_dist, 'ft = ',sig(min_C_L_for_takeoff, sf))

print('Minimum take-off distance = ',sig(min_dist,sf), ' ft')
print('C_L for minimum take-off = ',sig(C_L_min_dist,sf))
print('Clearance of 50 ft Obstacle at minimum take-off = ',sig(min_h_TR,sf), ' ft')
print('Transition Climb Angle = ',sig(min_gamma,sf), ' deg')
print('Take-off Distance at C_L = 3.18 = ',design_takeoff_dist)

plt.figure(figsize=(8,7))
plt.plot(C_L_array, take_off_array, label='Total Distance')
plt.plot(C_L_array, S_G_array, label='Ground Roll', color='b')
plt.plot(C_L_array, S_TR_array, label='Transition Distance', color='purple')
plt.plot(C_L_array, S_R_array, label='Rotation Ground Roll Distance',color='black')
plt.xlabel('$C_L (-)$')
plt.ylabel('Take-off Distance (ft)')
plt.grid()
plt.legend()
plt.axhline(y=5950,color='r',linestyle='--')
plt.annotate('5950 ft',[0.5, 6100])
plt.plot(C_L_min_dist, min_dist, marker='x', color='black',markersize=20)
if min_dist < runway_dist:
    plt.plot(min_C_L_for_takeoff, runway_dist, marker='x', color='black',markersize=20)



takeOff = Take_Off(T_max, W_max, S, mu, rho_sl, 3.18, C_D0, K)

S_G = takeOff[1]
V_to = takeOff[7]
S_R = takeOff[2]
S_TR = takeOff[3]
V_stall = takeOff[6]

C = 0.478 / 3600

# dS/dt = V -> dt = dS/V
# m / (m / s) = s
roll_time = S_G / (1 / np.sqrt(2) * V_to)
rotation_time = 3
S_TR_time = S_TR / (1.2*V_stall)

# C*T = kg/s * (kg*m/s^2) 

###
# Weights
###

fuel_mission_1 = 487919.0187




C_idle = 0.325*2.20462
test = Landing(W_max, fuel_mission_1, S, C_idle, rho_sl, 2.5, 0.06)


### fuel consumption
Wdot_takeoff = C*T_max*(1/np.sqrt(2))
roll_fuel = Wdot_takeoff*roll_time

rotation_fuel = rotation_time*T_max*C

S_TR_fuel = S_TR_time*T_max*C

take_off_fuel = roll_fuel + rotation_fuel + S_TR_fuel


# thrust at 70% V_to

# kg*m/s^2 -> mdot*V_to T = mdot*V_to 


### Climb



C_L = 1.5

rho_len = 10
rho_step = (rho_sl - rho_cruise) / rho_len
rho_array = np.arange(rho_cruise, rho_sl, rho_step)


V_stall_array = stall_vel(W_max, S, rho_array, C_L)
stall_index = np.zeros_like(rho_array)
V_v_max = np.zeros_like(rho_array)
V_v_index = np.zeros_like(rho_array)
V_best_climb = np.zeros_like(rho_array)


V_min = np.min(V_stall_array)
#V_min = 0
V_array = np.arange(V_min, V_max, step_size)


gamma_climb, V_v = np.zeros([rho_len, len(V_array)]), np.zeros([rho_len, len(V_array)])
D_array = np.zeros([rho_len, len(V_array)])

for i in range(len(V_array)):
    for j in range(rho_len):
        gamma_climb[j][i], V_v[j][i], D_array[j][i] = Climb(V_array[i], T_cont, W_max, rho_array[j], C_L)

V_v = V_v * 60  #ft/min
V_array = V_array*0.592484
V_stall_array = V_stall_array*0.592484

plt.figure(100)
plt.ylim(0, np.max(V_v)*1.05)
plt.xlim(V_min*0.592484,V_max*0.592484)
plt.grid()
plt.xlabel('Velocity (kts)')
plt.ylabel('Vertical Velocity (ft/min)')

for i in range(rho_len):

    stall_index[i] = np.where(V_array > V_stall_array[i])[0][0]
    index = int(stall_index[i])
    index = 0
    
    V_v_max[i] = np.max(V_v[i])
    V_v_index[i] = np.where(V_v_max[i] == V_v[i])[0][0]
    V_best_climb[i] = V_array[int(V_v_index[i])]
    
    plt.plot(V_array[index:], V_v[i][index:])
    plt.plot(V_best_climb[i], V_v_max[i], marker='x', color='black')
    


"""
V_v = V(T-D) = V*T - 1/2*rho*S*CD*V**3
T = D + Wsin -> (D - D + Wsin)
"""

# T_test = 2.805*2.20462*V*4

### Cruise ###
# L = W
# T = D




### Loiter ###

# Fuel consumption: C*T*(20*60) # for 20 minute loiter 

# fuel consumption in climb

V_v_avg = np.average(V_v_max)

V_v_index_upper = np.where(V_v_max > V_v_avg)[0][-1]
V_v_index_lower = V_v_index_upper + 1

time_to_climb = cruise_alt / V_v_avg # minutes

climb_fuel = time_to_climb*C*T_cont * 60



### landing
W_end = np.array([594740.95345023, 588437.47147406, 456424.30330433, 424722.93876948])
C_L_max_land = 3.18
mu_break = 0.3 #best conditions


W_max = 708700



best_landing = Landing(W_max, W_end[2], S, C_idle, rho_sl, C_L_max_land, mu)

print('Landing Distance = ',sig(best_landing[-1][0], 3))
print('Climb Rate = ',sig(V_v_avg, 3))

