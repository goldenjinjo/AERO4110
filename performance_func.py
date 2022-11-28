# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 23:50:10 2022

@author: usern
"""
import numpy as np
from variables import *
from sig import sig
import matplotlib.pyplot as plt

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
    gamma = np.arcsin( (T - D) / (W*g) )
    return gamma

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

    if h_TR.any() > h_obstacle:
        S_TR = np.sqrt(R_trans**2 - (R_trans - h_obstacle)**2)
    else:
        S_TR = R_trans*np.sin(gamma_climb)


    takeoff_distance = S_G + S_R + S_TR
    
    return takeoff_distance, S_G, S_R, S_TR, h_TR, R_trans, np.degrees(gamma_climb),  \
        V_stall, V_to, V_trans, K_T, K_A


def Landing(W_land, S, idle_flow, rho, C_L, mu):
    
    
    gamma_land = np.radians(3) # approach angle must be 3 degrees
    
    V_stall = stall_vel(W_land, S, rho, C_L)
    
    V_a = 1.2*V_stall
    
    D_land = Drag(rho, V_a, S, C_D0, K, C_L)
    
    T_land = np.sin(gamma_land)*W_land*g + D_land
    
    V_TD = 1.1*V_stall
    
    V_f = (V_a + V_TD) / 2
    
    R = V_f**2 / (0.2*g)
    
    h_TR = R*(1 - np.cos(gamma_land))
    
    S_a = (h_obstacle - h_TR)/np.tan(gamma_land)
    
    S_flare = R*np.sin(gamma_land)
    
    S_free_roll = 3*V_TD
    
    T_roll = (idle_flow * V_TD)*4
    
    roll = ground_dist(T_roll, W_land, S, mu, rho, C_L, C_D0, K, V_TD, 0)
    
    S_G, K_T, K_A = roll
    
    dist_tot = S_a + S_flare + S_free_roll + S_G
    
    return dist_tot, S_a, S_flare, S_free_roll, S_G, D_land, T_land, T_roll, \
        h_TR, R, V_stall, V_a, V_TD, V_f, K_T, K_A


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
    
    V_v = V*(T - D) / (W*g)
    
    return gamma, V_v, D
    

def C_L_cruise(W, rho, S, V):
    
    C_L = (2*W) / (rho * S * V**2)
    
    if metric:
        C_L = C_L*g
    
    return C_L


# numerical range eqn
def Range_Num(W_int, rho, V, step, distance_req):
    
    
    dist_travel = 0
    fuelLoss = 0
    time = 0
    
    
    C_L_int = C_L_cruise(W_int, rho, S, V)
        
    
    D_int = Drag(rho, V, S_ref, C_D0, K, C_L_int)
    
    W = np.zeros(int(1e6))
    W[0] = W_int
    
    C_L, D = np.zeros_like(W), np.zeros_like(W)
    
    C_L[0] = C_L_int
    D[0] = D_int
    
    fuelLoss_array = np.zeros_like(W)
    
    
    
    for i in range(1, len(W)):
        
        fuelLoss = -C*D[i-1]*step
        fuelLoss_array[i] = fuelLoss_array[i-1] + fuelLoss
        W[i] = W[i-1] + fuelLoss
        C_L[i] = C_L_cruise(W[i-1], rho, S, V)
        D[i] = Drag(rho, V, S_ref, C_D0, K, C_L[i-1])
        
        dist_travel += V*step
        time += step
        if dist_travel > distance_req:
            break
    
    fuelLoss_array[0] = fuelLoss_array[1]
    
    W = np.trim_zeros(W)
    C_L = np.trim_zeros(C_L)
    D = np.trim_zeros(D)
    fuelLoss_array = np.trim_zeros(fuelLoss_array)
    
    W_loss = W[0] - W[-1]
    #W_percent = W_loss / W[0]
    
    
    C_L_avg = np.average(C_L)
    
    # requirement to have extra fuel equal to 10% greater mission time
    extraFuel = C*np.average(D)*time*0.1
    
    return W_loss, time, dist_travel, extraFuel, W, C_L, D, fuelLoss_array, C_L_avg



### climbiter ###

def Climb_Optim(W, C_L, T=T_cont, rho_cruise=rho_cruise, cruise_alt=cruise_alt, iterations=10, step_size=step_size):
    """
    Climb Optimisation
    
    iterations : number of altitudes to plot

    Returns
    -------
    None.

    """
   
    rho_step = (rho_sl - rho_cruise) / iterations
    rho_array = np.arange(rho_cruise, rho_sl+rho_step, rho_step)
    
    alt_step = cruise_alt / iterations
    alt_array = np.arange(0, cruise_alt+alt_step, alt_step) * ft
    V_stall_array = stall_vel(W, S, rho_array, C_L)
    
    V_min = np.min(V_stall_array)
    V_array = np.arange(V_min, V_max, step_size)
    
    # initalising arrays
    stall_index = np.zeros_like(rho_array)
    V_v_max = np.zeros_like(rho_array)
    V_v_index = np.zeros_like(rho_array)
    V_best_climb = np.zeros_like(rho_array)
    gamma_climb, V_v = np.zeros([iterations+1, len(V_array)]), np.zeros([iterations+1, len(V_array)])
    D_array = np.zeros([iterations+1, len(V_array)])
    gamma_best_climb = np.zeros(iterations+1)
    
    for i in range(len(V_array)):
        for j in range(iterations):
            gamma_climb[j][i], V_v[j][i], D_array[j][i] = \
                Climb(V_array[i], T, W, rho_array[j], C_L)
                
            gamma_climb[j][i] = np.degrees(gamma_climb[j][i])
            V_v[j][i] = V_v[j][i]*ft*60
            D_array[j][i] = D_array[j][i]*lbf
            

    V_array = V_array*kts
    V_stall_array = V_stall_array*kts

    
    plt.figure(str(W))
    plt.title('$W_{to}$ = '+str(sig(W*lb,sf))+' $lb, C_L$ = '+str(C_L))
    plt.ylim(0, np.max(V_v)*1.05)
    plt.xlim(V_min*kts,V_max*kts)
    plt.grid()
    plt.xlabel('Velocity (kts)')
    plt.ylabel('Vertical Velocity (ft/min)')
    
    
    for i in range(iterations+1):
    
        alt_iter = np.int ( np.round( alt_array[i], 0) )
    
        
        stall_index[i] = np.where(V_array > V_stall_array[i])[0][0]
        index = int(stall_index[i])
        index = 0
        
        V_v_max[i] = np.max(V_v[i])
        V_v_index[i] = np.where(V_v_max[i] == V_v[i])[0][0]
        V_best_climb[i] = V_array[int(V_v_index[i])]
        gamma_best_climb[i] = gamma_climb[i][int(V_v_index[i])]
        
        plt.plot(V_array[index:], V_v[i][index:], label=str(alt_iter) )
        plt.plot(V_best_climb[i], V_v_max[i], marker='x', color='black')
        
    plt.legend(title='Altitude (ft)')
    V_v_avg = np.mean(V_v_max)
    
    gamma_best_climb_avg = np.mean(gamma_best_climb)


    
    V_avg = np.mean(V_best_climb)
    
    return V_v_avg, gamma_best_climb_avg, V_avg, V_v, gamma_climb, D_array, gamma_best_climb