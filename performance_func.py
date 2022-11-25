# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 23:50:10 2022

@author: usern
"""
import numpy as np
from variables import *

def stall_vel(W, S,rho, C_L):
    
    if metric:
        V = np.sqrt( (2*W*g) / (rho*S*C_L) )
    else:
        V = np.sqrt( (2*W) / (rho*S*C_L) )
    
    return V

def Lift(rho, V, C_L, S):
    return 1/2 * rho * C_L * S * V**2

def Drag(rho, V, S, C_D0, K, C_L):
    return 1/2*rho*V**2*S* (C_D0 + K*C_L**2)




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


def min_vel(W, rho, S, K, C_D0):
    return np.sqrt( (2*W) / (rho*S) * np.sqrt(K/C_D0))


# Equation 17.3 Raymer
def Endurance(L, D, C, W_i, W_f):
    return (L / D) * (1 / C) * np.log(W_i / W_f)

# Equation  17.23, Raymer
def Range(L, D, C, W_i, W_f, V):
    return Endurance(L,D,C,W_i,W_f) * V

### best range equations
def Range_Parameters(W):
    
    V = np.sqrt((2*W)/(rho_cruise*S) * np.sqrt(3*K / C_D0))
    D = 1/2 * rho_cruise * V**2 * S * (4/3 * C_D0)
    C_L = np.sqrt(C_D0 / (3*K))
    
    return V, D, C_L


def Take_Off(T, W, S, mu, rho, C_L, C_D0, K, n=1.2):
    
    V_stall = stall_vel(W, S, rho, C_L)
    V_to = 1.1*V_stall
    V_trans = 1.15*V_stall
    D_to = Drag(rho_sl, V_to, S, C_D0, K, C_L)
    
    S_G, K_T, K_A = ground_dist(T, W, S, mu, rho, C_L, C_D0, K, 0, V_to)
    S_R = 3*V_to
    
    R_trans = V_trans**2 / (g * (n - 1))

    gamma_climb = Gamma(T, D_to, W)

    h_TR = R_trans*(1 - np.cos(gamma_climb))

    if h_TR > h_obstacle:
        S_TR = np.sqrt(R_trans**2 - (R_trans - h_obstacle)**2)
    else:
        S_TR = R_trans*np.sin(gamma_climb)


    takeoff_distance = S_G + S_R + S_TR
    
    return takeoff_distance, S_G, S_R, S_TR, h_TR, gamma_climb, V_stall, V_to, V_trans, K_T, K_A


def Landing(W_to, W_land, S, C_idle, rho, C_L, mu):
    
    # Leave 20 percent of reserves
    #W_land = 0.85*W_to
    
    V_stall = stall_vel(W_land, S, rho, C_L)
    
    V_a = 1.2*V_stall
    
    T_idle = (C_idle * V_a)*4 * 10
    
    D_land = Drag(rho, V_a, S, C_D0, K, C_L)
    
    gamma_land = Gamma(T_idle, D_land, W_land)
    
    
    V_TD = 1.1*V_stall
    
    V_f = (V_a + V_TD) / 2
    
    R = V_f**2 / (0.2*g)
    
    S_flare = R*np.sin(gamma_land)
    
    S_free_roll = 3*V_TD
    
    T_roll = (C_idle * V_TD)*4
    
    S_G = ground_dist(T_idle, W_land, S, mu, rho, C_L, C_D0, K, V_TD, 0)
    
    dist_tot = S_flare + S_free_roll + S_G
    
    return V_a, D_land, T_idle, np.degrees(gamma_land), S_flare, S_free_roll, S_G, dist_tot


def Climb(V, T, W, rho, C_L):
    
    C_D = C_D0 + K*C_L**2
    
    D = 1/2 * rho * V**2 * S * C_D
    
    gamma = Gamma(T,D,W)
    
    # V = (2 * W) / (rho*C_L*S) * np.cos(gamma)
    # gamma*W = T - D, T = gamma*W + D
    # T = D + Wsin(gamma)
    # D = T - Wsin(gamma)
    # V^2 = 2*(T - Wsin(gamma))/(rho*S*C_D)
    # V = np.sqrt(2*(T - W*np.sin(gamma)) / (rho*S*C_D) )
    
    
    #V = np.sqrt(2*(T - W*np.sin(gamma)) / (rho*S*C_D) )
    
    V_v = V*(T - D) / W
    
    return gamma, V_v, D
    