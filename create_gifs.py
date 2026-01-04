import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import io
from scipy.integrate import solve_ivp

plt.ioff()

def CreateBallDropGif():
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

    Y0 = [0.1461, 0, 0.1161, 0]
    TStart = 0
    TEnd = 0.7
    
    TimeData = []
    M1HeightData = []
    M2HeightData = []
    
    while TStart < TEnd:
        SolFall = solve_ivp(FreeFall, [TStart, TEnd], Y0, max_step=MaxStep, events=TerminateGround)
        TimeData.extend(SolFall.t)
        M1HeightData.extend(SolFall.y[0])
        M2HeightData.extend(SolFall.y[2])
        Y0 = [SolFall.y[0][-1], SolFall.y[1][-1], 0, 0]
        SolCompress = solve_ivp(Compression, [SolFall.t[-1], SolFall.t[-1] + 1], Y0, max_step=MaxStep, events=TerminateSpring)
        TimeData.extend(SolCompress.t)
        M1HeightData.extend(SolCompress.y[0])
        M2HeightData.extend(SolCompress.y[2])
        Y0 = [SolCompress.y[0][-1], SolCompress.y[1][-1], SolCompress.y[2][-1], SolCompress.y[3][-1]]
        SolRebound = solve_ivp(Rebound, [SolCompress.t[-1], SolCompress.t[-1] + 2], Y0, max_step=MaxStep, events=TerminateRebound)
        TimeData.extend(SolRebound.t)
        M1HeightData.extend(SolRebound.y[0])
        M2HeightData.extend(SolRebound.y[2])
        TStart = SolRebound.t[-1]
        Y0 = [SolRebound.y[0][-1], SolRebound.y[1][-1], SolRebound.y[2][-1], SolRebound.y[1][-1]]
        if len(TimeData) > 500:
            break

    Frames = []
    Skip = max(1, len(TimeData) // 80)
    for i in range(0, len(TimeData), Skip):
        Fig, Ax = plt.subplots(figsize=(6, 4))
        Ax.plot(TimeData[:i+1], np.array(M1HeightData[:i+1]) - L0, 'b-', linewidth=2, label='M1')
        Ax.plot(TimeData[:i+1], np.array(M2HeightData[:i+1]), 'r-', linewidth=2, label='M2')
        Ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        Ax.plot(TimeData[i], M1HeightData[i] - L0, 'bo', markersize=8)
        Ax.plot(TimeData[i], M2HeightData[i], 'ro', markersize=8)
        Ax.set_xlabel('t')
        Ax.set_ylabel('h')
        Ax.set_title('Ball Drop')
        Ax.legend()
        Ax.grid(True, alpha=0.3)
        plt.tight_layout()
        Buf = io.BytesIO()
        Fig.savefig(Buf, format='png', dpi=80, bbox_inches='tight')
        Buf.seek(0)
        Img = Image.open(Buf).resize((600, 400))
        Frames.append(np.array(Img))
        plt.close(Fig)
    
    imageio.mimsave('ball_drop.gif', Frames, fps=10, loop=0)

def CreateSlipJumpGif():
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

    def Flight(_, Y):
        return [Y[1], 0, Y[3], -G]

    def TerminateFlight(_, Y):
        return R0*np.sin(Theta0) - Y[2]

    TerminateStance1.terminal = True
    TerminateFlight.terminal = True
    TerminateFlight.direction = 1

    SolStance1 = solve_ivp(Stance, [0, 1], [1.1*np.pi/2, -0.5, 0.999, -2], max_step=MaxStep, events=TerminateStance1)
    SolFlight1 = solve_ivp(Flight, [0, 1], [SolStance1.y[2][-1]*np.cos(SolStance1.y[0][-1]), SolStance1.y[3][-1]*np.cos(SolStance1.y[0][-1]) - SolStance1.y[2][-1]*SolStance1.y[1][-1]*np.sin(SolStance1.y[0][-1]), SolStance1.y[2][-1]*np.sin(SolStance1.y[0][-1]), SolStance1.y[3][-1]*np.sin(SolStance1.y[0][-1]) + SolStance1.y[2][-1]*SolStance1.y[1][-1]*np.cos(SolStance1.y[0][-1])], max_step=MaxStep, events=TerminateFlight)
    Distance = SolFlight1.y[0][-1]-R0*np.cos(Theta0)
    SolStance2 = solve_ivp(Stance, [0, 1], [Theta0, -(SolFlight1.y[1][-1]*np.cos(np.arctan2(Distance-SolFlight1.y[0][-1], SolFlight1.y[2][-1])) + SolFlight1.y[3][-1]*np.sin(np.arctan2(Distance-SolFlight1.y[0][-1], SolFlight1.y[2][-1])))/R0, R0*0.99, (-SolFlight1.y[1][-1]*np.sin(np.arctan2(Distance-SolFlight1.y[0][-1], SolFlight1.y[2][-1])) + SolFlight1.y[3][-1]*np.cos(np.arctan2(Distance-SolFlight1.y[0][-1], SolFlight1.y[2][-1])))], max_step=MaxStep, events=TerminateStance1)
    SolFlight2 = solve_ivp(Flight, [0, 1], [Distance+SolStance2.y[2][-1]*np.cos(SolStance2.y[0][-1]), SolStance2.y[3][-1]*np.cos(SolStance2.y[0][-1]) - SolStance2.y[2][-1]*SolStance2.y[1][-1]*np.sin(SolStance2.y[0][-1]), SolStance2.y[2][-1]*np.sin(SolStance2.y[0][-1]), SolStance2.y[3][-1]*np.sin(SolStance2.y[0][-1]) + SolStance2.y[2][-1]*SolStance2.y[1][-1]*np.cos(SolStance2.y[0][-1])], max_step=MaxStep, events=TerminateFlight)

    XStance1 = SolStance1.y[2] * np.cos(SolStance1.y[0])
    YStance1 = SolStance1.y[2] * np.sin(SolStance1.y[0])
    XFlight1 = SolFlight1.y[0]
    YFlight1 = SolFlight1.y[2]
    XStance2 = Distance + SolStance2.y[2] * np.cos(SolStance2.y[0])
    YStance2 = SolStance2.y[2] * np.sin(SolStance2.y[0])
    XFlight2 = SolFlight2.y[0]
    YFlight2 = SolFlight2.y[2]

    Frames = []
    Skip = max(1, len(XStance1) // 30)
    for i in range(0, len(XStance1), Skip):
        Fig, Ax = plt.subplots(figsize=(6, 4))
        Ax.plot([-0.25, 1.6], [0, 0], 'k--', linewidth=1)
        Ax.plot([0, XStance1[i]], [0, YStance1[i]], 'b-', linewidth=2)
        Ax.plot(XStance1[i], YStance1[i], 'bo', markersize=8)
        Ax.set_xlabel('X')
        Ax.set_ylabel('Y')
        Ax.set_title('SLIP Jump')
        Ax.axis('equal')
        Ax.grid(True, alpha=0.3)
        plt.tight_layout()
        Buf = io.BytesIO()
        Fig.savefig(Buf, format='png', dpi=80, bbox_inches='tight')
        Buf.seek(0)
        Img = Image.open(Buf).resize((600, 400))
        Frames.append(np.array(Img))
        plt.close(Fig)

    Skip = max(1, len(XFlight1) // 30)
    for i in range(0, len(XFlight1), Skip):
        Fig, Ax = plt.subplots(figsize=(6, 4))
        Ax.plot([-0.25, 1.6], [0, 0], 'k--', linewidth=1)
        Ax.plot(XFlight1[:i+1], YFlight1[:i+1], 'r-', linewidth=2)
        Ax.plot(XFlight1[i], YFlight1[i], 'ro', markersize=8)
        Ax.set_xlabel('X')
        Ax.set_ylabel('Y')
        Ax.set_title('SLIP Jump')
        Ax.axis('equal')
        Ax.grid(True, alpha=0.3)
        plt.tight_layout()
        Buf = io.BytesIO()
        Fig.savefig(Buf, format='png', dpi=80, bbox_inches='tight')
        Buf.seek(0)
        Img = Image.open(Buf).resize((600, 400))
        Frames.append(np.array(Img))
        plt.close(Fig)

    Skip = max(1, len(XStance2) // 30)
    for i in range(0, len(XStance2), Skip):
        Fig, Ax = plt.subplots(figsize=(6, 4))
        Ax.plot([-0.25, 1.6], [0, 0], 'k--', linewidth=1)
        Ax.plot(XFlight1, YFlight1, 'r-', linewidth=1, alpha=0.3)
        Ax.plot([Distance, XStance2[i]], [0, YStance2[i]], 'b-', linewidth=2)
        Ax.plot(XStance2[i], YStance2[i], 'bo', markersize=8)
        Ax.set_xlabel('X')
        Ax.set_ylabel('Y')
        Ax.set_title('SLIP Jump')
        Ax.axis('equal')
        Ax.grid(True, alpha=0.3)
        plt.tight_layout()
        Buf = io.BytesIO()
        Fig.savefig(Buf, format='png', dpi=80, bbox_inches='tight')
        Buf.seek(0)
        Img = Image.open(Buf).resize((600, 400))
        Frames.append(np.array(Img))
        plt.close(Fig)

    Skip = max(1, len(XFlight2) // 30)
    for i in range(0, len(XFlight2), Skip):
        Fig, Ax = plt.subplots(figsize=(6, 4))
        Ax.plot([-0.25, 1.6], [0, 0], 'k--', linewidth=1)
        Ax.plot(XFlight1, YFlight1, 'r-', linewidth=1, alpha=0.3)
        Ax.plot(XStance2, YStance2, 'b-', linewidth=1, alpha=0.3)
        Ax.plot(XFlight2[:i+1], YFlight2[:i+1], 'r-', linewidth=2)
        Ax.plot(XFlight2[i], YFlight2[i], 'ro', markersize=8)
        Ax.set_xlabel('X')
        Ax.set_ylabel('Y')
        Ax.set_title('SLIP Jump')
        Ax.axis('equal')
        Ax.grid(True, alpha=0.3)
        plt.tight_layout()
        Buf = io.BytesIO()
        Fig.savefig(Buf, format='png', dpi=80, bbox_inches='tight')
        Buf.seek(0)
        Img = Image.open(Buf).resize((600, 400))
        Frames.append(np.array(Img))
        plt.close(Fig)
    
    imageio.mimsave('slip_jump.gif', Frames, fps=10, loop=0)

if __name__ == "__main__":
    CreateBallDropGif()
    CreateSlipJumpGif()

