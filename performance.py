# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:49:13 2022


AERO4110 Code - Performance and Flight Dynamics

@author: usern
"""
import numpy as np
import math
import matplotlib.pyplot as plt


"""
Variables
"""
g = 32.174 #ft/s^2
S = 6900 #ft^2
W_max = 1030500 # lb
rho_sl = 0.002377693 #slug/ft^3
K = 0.055579171
C_L_max = 3.2 #2.7198
h_obstacle = 50 #ft
T_max = 388000*(3/4) # take off thrust
mu = 0.05 #rolling resistance
C_D0 = 0.015 #historical


"""
Functions
"""

def stall_vel(W, S,rho, C_L):
    return np.sqrt( (2*W_max) / (rho*S*C_L) )

def Lift(rho, V, C_L, S):
    return 1/2 * rho * C_L * S * V**2

def Drag(rho, V, S, C_D0, K, C_L):
    return 1/2*rho*V**2* (C_D0 + K*C_L**2)


# Equation 17.3 Raymer
def Endurance(L, D, C, W_i, W_f):
    return (L / D) * (1 / C) * np.log(W_i / W_f)

# Equation  17.23, Raymer
def Range(L, D, C, W_i, W_f, V):
    return Endurance(L,D,C,W_i,W_f) * V


### GROUND ROLL
def ground_accel(W, T, D, L, mu):
    return g / W * (T - D - mu*(W - L) )

def ground_dist(T, W, S, mu, rho, C_L, C_D0, K, V_I, V_F):
    
    K_T = (T / W) - mu
    K_A = (rho * ( mu*C_L - C_D0 - K*C_L**2 ) ) / ( 2 * ( W / S ) )
    
    return (1 / (2*g*K_A) )*np.log((K_T + K_A*V_F**2 ) / (K_T + K_A*V_I**2) ), K_T, K_A

# Transition
def Gamma(T, D, W):
    return np.arcsin( (T - D) / W )



### Steady, Level Flight. Cruise.

# Eq 17.10 Raymer
# Use this to determine optimal Cruise altitude
def cruise_vel(rho, C_L, W, S):
    return np.sqrt( (2 * W) / (rho*C_L * S))

def thrust_to_weight(q, C_D0, W, S, K):
    return q*C_D0 / (W / S) + (W / S) * (K / q)

def W_dot(C, T):
    return -C*T


## Fuel Required:
# C * T at the different stages of flight.
# make sure range requirement is met.


def Take_Off(T, W, S, mu, rho, C_L, C_D0, K, n=1.2):
    
    V_stall = stall_vel(W, S, rho, C_L)
    V_to = 1.1*V_stall
    V_trans = 1.15*V_stall
    
    S_G, K_T, K_A = ground_dist(T, W, S, mu, rho, C_L, C_D0, K, 0, V_to)
    S_R = 3*V_to
    
    R_trans = V_trans**2 / (g * (n - 1))

    gamma_climb = Gamma(T_max*0.8, D_to, W_max)

    h_TR = R_trans*(1 - math.cos(gamma_climb))

    if h_TR.any() > h_obstacle:
        S_TR = np.sqrt(R_trans**2 - (R_trans - h_obstacle)**2)
    else:
        S_TR = R_trans*math.sin(gamma_climb)


    takeoff_distance = S_G + S_R + S_TR
    
    return takeoff_distance, S_G, S_R, S_TR, h_TR, gamma_climb, V_stall, V_to, V_trans, K_T, K_A



"""
Computation
"""

V_stall = stall_vel(W_max, S, rho_sl, C_L_max)
V_to = 1.1*V_stall

L_to = Lift(rho_sl, V_to, C_L_max, S) # lift at take-off
D_to = Drag(rho_sl, V_to, S, C_D0, K, C_L_max) # drag at take-off


C_L_array = np.arange(2.5, 3.7, 0.1)
K_A_array = np.zeros_like(C_L_array)
K_T_array = np.zeros_like(C_L_array)
V_to_array = np.zeros_like(C_L_array)
gamma_array = np.zeros_like(C_L_array)
S = 7780
for i in range(len(C_L_array)):
        K_A_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[-1]
        K_T_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[-2]
        V_to_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[-4]
        gamma_array[i] = Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array[i], C_D0, K)[5]

plt.plot(C_L_array, Take_Off(T_max, W_max, S, mu, rho_sl, C_L_array, C_D0, K)[0])
plt.xlabel('$C_L (-)$')
plt.ylabel('Take-off Distance (ft)')
plt.title('S = '+str(S)+' $ft^2$')
plt.grid()
#plt.plot(C_L_array, np.ones(len(C_L_array))*7000)
plt.axhline(y=7000,color='r',linestyle='--')
plt.annotate('7000 ft',[0.5, 7300])

#test = Take_Off(T_max, W_max, S, mu, rho_sl, 3, C_D0, K)

test = K_T_array + K_A_array*V_to_array**2