# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:14:51 2022

@author: usern
"""
import numpy as np
class Class:
    
    testVar = 5
    
    def __init__(self, name):
        self.get_logger()
        
        pass
    
test = {1: 'a',
        2: 'b'}

test[1] = 'c'

def func(a):
    if a == 1:
        raise Exception('no')
    return a + 2


test2 = {0: 'TEMP', 1: 10, 2: 40}

test3 = {0: 'TEMP2', 1: 30, 2: 40}

sensorName = np.array(['TEMP', 'TEMP2', 'TEMP3'])

q = [1, 2]
w = [5, 53]
e = [122, 7]

array = [q,w,e]

test_dict = {}

for i in range(len(sensorName)):
    test_dict[sensorName[i]] = array[i]

test4 = {'TEMP': [1,2], 'b': [5253, 342432]}
test5 = {'TEMP2': [3,4]}

aa, bb = test4['TEMP']

a = np.array([test4, test5])

for i in range(len(a)):
   for j in sensorName:
       pass
        
   
k = 5

k -= 1

kk = [1,2,3]

kk.append(5)

sensorName = np.append(sensorName, 5)

ii = np.array([2])

ii = np.append(ii, ii+0.000001)

ii = np.append(ii, ii+0.0000001)
