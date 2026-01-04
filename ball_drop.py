import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

M1 = 1.052
M2 = 0.060
G = 9.81
K = 9300
L0 = 0.03
MaxStep = 0.001

C0 = 2
C1 = 0.2
C2 = 10
C3 = 6.1

def FreeFall(_, Y):
    DampingM1 = -C0 * Y[1]
    DampingM2 = -C1 * Y[3]
    FSpring = K * max(L0 - (Y[0] - Y[2]), 0)
    return [Y[1], -G + DampingM1 / M1 + FSpring / M1, Y[3], -G + DampingM2 / M2]

def Compression(_, Y):
    FSpring = K * max(L0 - (Y[0] - Y[2]), 0)
    DampingM1 = -C2 * Y[1]
    return [Y[1], -G + FSpring / M1 + DampingM1 / M1, 0, 0]

def Rebound(_, Y):
    DampingM = -C3 * Y[1]
    return [Y[1], -G + DampingM / (M1+M2), Y[1], -G + DampingM / (M1+M2)]

def TerminateGround(_, Y):
    return Y[2] if Y[1] < 0 else 1

def TerminateSpring(_, Y):
    return 0 if Y[0] > L0 and Y[1] > 0 else 1

def TerminateRebound(_, Y):
    return 0 if Y[1] < 0 else 1

TerminateGround.terminal = True
TerminateGround.direction = -1
TerminateSpring.terminal = True
TerminateSpring.direction = 1
TerminateRebound.terminal = True
TerminateRebound.direction = -1

PlotEvery = 3

Fig = plt.figure(figsize=[4, 3])
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('t')
plt.ylabel('h')
plt.tight_layout()

Y0 = [0.1461, 0, 0.1161, 0]
TStart = 0
TEnd = 0.7

TimeData = []
M1HeightData = []
M1VelocityData = []
M2HeightData = []
M2VelocityData = []

while TStart < TEnd:
    SolFall = solve_ivp(FreeFall, [TStart, TEnd], Y0, max_step=MaxStep, events=TerminateGround)
    TimeData.extend(SolFall.t)
    M1HeightData.extend(SolFall.y[0])
    M1VelocityData.extend(SolFall.y[1])
    M2HeightData.extend(SolFall.y[2])
    M2VelocityData.extend(SolFall.y[3])
    
    for i in range(len(SolFall.t)):
        if i % PlotEvery == 0:
            plt.plot(SolFall.t[i], SolFall.y[0][i] - L0, 'bo', alpha=i/len(SolFall.t))
            plt.plot(SolFall.t[i], SolFall.y[2][i], 'ro', alpha=i/len(SolFall.t))
            plt.pause(1e-5)
    
    if TStart >= TEnd:
        break
    
    Y0 = [SolFall.y[0][-1], SolFall.y[1][-1], 0, 0]
    SolCompress = solve_ivp(Compression, [SolFall.t[-1], SolFall.t[-1] + 1], Y0, max_step=MaxStep, events=TerminateSpring)
    TimeData.extend(SolCompress.t)
    M1HeightData.extend(SolCompress.y[0])
    M1VelocityData.extend(SolCompress.y[1])
    M2HeightData.extend(SolCompress.y[2])
    M2VelocityData.extend(SolCompress.y[3])
    
    for i in range(len(SolCompress.t)):
        if i % PlotEvery == 0:
            plt.plot(SolCompress.t[i], SolCompress.y[0][i] - L0, 'bo', alpha=i/len(SolCompress.t))
            plt.plot(SolCompress.t[i], SolCompress.y[2][i], 'ro', alpha=i/len(SolCompress.t))
            plt.pause(1e-5)
    
    if TStart >= TEnd:
        break
    
    Y0 = [SolCompress.y[0][-1], SolCompress.y[1][-1], SolCompress.y[2][-1], SolCompress.y[3][-1]]
    SolRebound = solve_ivp(Rebound, [SolCompress.t[-1], SolCompress.t[-1] + 2], Y0, max_step=MaxStep, events=TerminateRebound)
    TimeData.extend(SolRebound.t)
    M1HeightData.extend(SolRebound.y[0])
    M1VelocityData.extend(SolRebound.y[1])
    M2HeightData.extend(SolRebound.y[2])
    M2VelocityData.extend(SolRebound.y[1])
    
    for i in range(len(SolRebound.t)):
        if i % PlotEvery == 0:
            plt.plot(SolRebound.t[i], SolRebound.y[0][i] - L0, 'bo', alpha=i/len(SolRebound.t))
            plt.plot(SolRebound.t[i], SolRebound.y[2][i], 'ro', alpha=i/len(SolRebound.t))
            plt.pause(1e-5)
    
    TStart = SolRebound.t[-1]
    Y0 = [SolRebound.y[0][-1], SolRebound.y[1][-1], SolRebound.y[2][-1], SolRebound.y[1][-1]]

plt.tight_layout()
plt.savefig('ball_drop.pdf')
