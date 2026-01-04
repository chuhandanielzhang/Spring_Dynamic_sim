import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

M = 1
G = 9.81
R0 = 1
MaxStep = 0.01
Theta0 = np.pi/2*1.2
K = 60

def Stance(_, Y):
    Ddtheta = -2 * Y[1] * Y[3] / Y[2] - G * np.cos(Y[0])/Y[2]
    Ddr = Y[1]**2*Y[2]-G*np.sin(Y[0])-K/M*(Y[2]-R0)
    return [Y[1], Ddtheta, Y[3], Ddr]

def TerminateStance1(_, Y):
    return R0 - Y[2]

def TerminateStance2(_, Y):
    return R0 - Y[2]

def Flight(_, Y):
    return [Y[1], 0, Y[3], -G]

def TerminateFlight(_, Y):
    return R0*np.sin(Theta0) - Y[2]

TerminateStance1.terminal = True
TerminateStance2.terminal = True
TerminateFlight.terminal = True
TerminateFlight.direction = 1
Skip = 3

Fig = plt.figure(figsize=[4, 3])
plt.plot(0, 0, 'ko')
plt.plot([-0.25, 1.6], [0, 0], 'k--')
plt.plot([-0.25, 1.6], [R0*np.sin(Theta0), R0*np.sin(Theta0)], 'k:')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.tight_layout()

Distance = 0
SolStance1 = solve_ivp(Stance, [0, 1], [1.1*np.pi/2, -0.5, 0.999, -2], max_step=MaxStep, events=TerminateStance1)
for i in range(len(SolStance1.t)):
    if i % Skip == 0:
        plt.plot([0, SolStance1.y[2][i]*np.cos(SolStance1.y[0][i])], [0, SolStance1.y[2][i]*np.sin(SolStance1.y[0][i])], 'b', alpha=i/len(SolStance1.t))
        plt.plot(SolStance1.y[2][i]*np.cos(SolStance1.y[0][i]), SolStance1.y[2][i]*np.sin(SolStance1.y[0][i]), 'bo', alpha=i/len(SolStance1.t))
        plt.pause(1e-5)

SolFlight1 = solve_ivp(Flight, [0, 1], [SolStance1.y[2][-1]*np.cos(SolStance1.y[0][-1]), SolStance1.y[3][-1]*np.cos(SolStance1.y[0][-1]) - SolStance1.y[2][-1]*SolStance1.y[1][-1]*np.sin(SolStance1.y[0][-1]), SolStance1.y[2][-1]*np.sin(SolStance1.y[0][-1]), SolStance1.y[3][-1]*np.sin(SolStance1.y[0][-1]) + SolStance1.y[2][-1]*SolStance1.y[1][-1]*np.cos(SolStance1.y[0][-1])], max_step=MaxStep, events=TerminateFlight)
for i in range(len(SolFlight1.t)):
    if i % Skip == 0:
        plt.plot(SolFlight1.y[0][i], SolFlight1.y[2][i], 'ro', alpha=i/len(SolFlight1.t))
        plt.tight_layout()
        plt.pause(1e-5)

Distance = SolFlight1.y[0][-1]-R0*np.cos(Theta0)
plt.plot(Distance, 0, 'ko')
SolStance2 = solve_ivp(Stance, [0, 1], [Theta0, -(SolFlight1.y[1][-1]*np.cos(np.arctan2(Distance-SolFlight1.y[0][-1], SolFlight1.y[2][-1])) + SolFlight1.y[3][-1]*np.sin(np.arctan2(Distance-SolFlight1.y[0][-1], SolFlight1.y[2][-1])))/R0, R0*0.99, (-SolFlight1.y[1][-1]*np.sin(np.arctan2(Distance-SolFlight1.y[0][-1], SolFlight1.y[2][-1])) + SolFlight1.y[3][-1]*np.cos(np.arctan2(Distance-SolFlight1.y[0][-1], SolFlight1.y[2][-1])))], max_step=MaxStep, events=TerminateStance1)
for i in range(len(SolStance2.t)):
    if i % Skip == 0:
        plt.plot([Distance, Distance+SolStance2.y[2][i]*np.cos(SolStance2.y[0][i])], [0, SolStance2.y[2][i]*np.sin(SolStance2.y[0][i])], 'b', alpha=i/len(SolStance2.t))
        plt.plot(Distance+SolStance2.y[2][i]*np.cos(SolStance2.y[0][i]), SolStance2.y[2][i]*np.sin(SolStance2.y[0][i]), 'bo', alpha=i/len(SolStance2.t))
        plt.pause(1e-5)

SolFlight2 = solve_ivp(Flight, [0, 1], [Distance+SolStance2.y[2][-1]*np.cos(SolStance2.y[0][-1]), SolStance2.y[3][-1]*np.cos(SolStance2.y[0][-1]) - SolStance2.y[2][-1]*SolStance2.y[1][-1]*np.sin(SolStance2.y[0][-1]), SolStance2.y[2][-1]*np.sin(SolStance2.y[0][-1]), SolStance2.y[3][-1]*np.sin(SolStance2.y[0][-1]) + SolStance2.y[2][-1]*SolStance2.y[1][-1]*np.cos(SolStance2.y[0][-1])], max_step=MaxStep, events=TerminateFlight)
for i in range(len(SolFlight2.t)):
    if i % Skip == 0:
        plt.plot(SolFlight2.y[0][i], SolFlight2.y[2][i], 'ro', alpha=i/len(SolFlight2.t))
        plt.tight_layout()
        plt.pause(1e-5)

plt.tight_layout()
plt.savefig('slip_jump.pdf')
