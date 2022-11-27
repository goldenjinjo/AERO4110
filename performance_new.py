#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 10:05:36 2022

@author: jinjo
"""

from variables import *
from performance_func import *
from sig import sig

run_cruise = False #save sanity



takeoff = Take_Off(T_max, W_to, S_ref, mu, rho_sl, C_L_max, C_D0, K)

takeoff_dist = takeoff[0]*ft
ground_roll = takeoff[1]*ft
ground_rot = takeoff[2]*ft



land = Landing(W_land, S_ref, idle_flow, rho_sl, C_L_max, mu)


distance_req_array = np.array([4800, 4400, 3800, 7000]) * 1 / nmi #meters




if run_cruise:
    
    cruise_drag = np.zeros(len(W_to))
    cruise_time = np.zeros(len(W_to))
    cruise_C_L = np.zeros(len(W_to))

    
    for i in range(len(W_to)):
        
        rho_temp = rho_cruise
        
        if i == 1:
            rho_temp = rho_28k
        cruise = Range_Num(W_to[i] - fuel_climb_estimate, rho_temp, V_max, 0.1, distance_req_array[i])
        
        cruise_drag[i] = sig(np.mean(cruise[-3])*lbf, sf)
        cruise_time[i] = sig(cruise[1] / 3600, sf)
        cruise_C_L[i] = sig(cruise[-1], sf)
    
        rho_temp = rho_cruise



## CLIMB
#1/2*rho*V**2*S*(C_D0+K*C_L**2) = T_cont
C_L_climb_lim = np.floor( np.sqrt( 1/K * ( (2*T_cont) / (rho_sl*V_max**2*S) - C_D0 ) ) )

C_L_avg = 1.65

V_v = np.zeros(len(W_to))
gamma_climb = np.zeros(len(W_to))
V_climb = np.zeros(len(W_to))

for i in range(len(W_to)):
    climb = Climb_Optim(W_to[i], C_L_avg)
    V_v[i] = climb[0]
    gamma_climb[i] = climb[1]
    V_climb[i] = climb[2]
    
    
# climb fuel consumption

time_climb = cruise_alt / (V_v / (60*ft))

W_loss_climb = time_climb * C * T_cont