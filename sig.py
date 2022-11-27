# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 05:59:44 2020

Very simple function that calculates significant figures of normal sized numbers

@author: Steven Watts
"""
import math
import numpy as np

# Dictionary to be used to scale numbers in correspondance with SI standard
SI_prefix_dict = {
'barn': -28,
'yocto': -24,
'zepto': -21,
'atto': -18,
'femto': -15,
'pico': -12,
'nano': -9,
'micro': -6,
'milli': -3,
'centi': -2,
'deci': -1,
'deca': 1,
'hecto': 2,
'kilo': 3,
'mega': 6,
'giga': 9,
'tera': 12,
'peta': 15,
'exa': 18,
'zetta': 21,
'yotta': 24
}


def sig(realNum, numSigFig, scale=0):
    """ 
    Returns given real number with the desired number of significant figures \n
    realNum : real number \n
    numSigFig : number of significant figures \n
    scale (default = 0) : scales real number by a factor of 10^(-3*scale).
            Input can be str in form of SI prefix e.g. 'kilo'.
    """
    
    
    # TODO add new check that is compatiable with numpy arrays
    """
    #if math.isnan(realNum).any() == True:
    #    return realNum

    if realNum==0:
        return 0
    """
    
    if numSigFig < 0:
        raise Exception('Error: Number of Significant Figures Cannot be Negative')
    
    
    # Optional scaling feature for realNum. Scales based on metric prefix.
    # e.g. for kilo, scale = 1.
    if type(scale) == int and scale != 0:
        realNum = realNum * 10**(3 * -scale)
    
    elif type(scale) == str:
        try:
            realNum = realNum * 10**(-SI_prefix_dict[scale])
        except KeyError:
            print('Scale Error: Undefined SI Prefix')
    
    elif type(scale) == float:
        print('Scale Error: Scale must be an Integer')
    
    
    
    # Uses the decadic logarithm to determine the number of zeros of realNum.
    # If the magnitude is negative, than the input number is between -1 and 1.
    magnitude = np.log10(abs(realNum))
    
    
    if magnitude.any() > 0:  
        # changes where round function operates, so it operates at the first
        # number rather than at the decimal point.
        res = np.round(realNum, -np.int(magnitude) + numSigFig - 1)
        
    else:
        # when magnitude is negative, the number is less than 1,
        # the round function needs no correction.! incorrect explanation.
        res = np.round(realNum, -np.int(magnitude) + numSigFig)
  
    
    # Formats the result as an int if no decimal places are calculated.
    if magnitude.any() >= 1 and \
    numSigFig <= magnitude + 1:
        res = np.int(res)
    
    return res
