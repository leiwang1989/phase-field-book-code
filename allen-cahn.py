import numpy as np
import matplotlib.pyplot as plt

nx = ny = 100
gamma = 0.1 # J/m2
W = 2.0e-9  # m 
eps = np.sqrt(W*gamma)
H = 18.0 * gamma/W
M_phi = 1.0 
dx = W/5.0 # 界面内放5个网格
dt = 0.8*dx*dx/(4.0*M_phi*eps*eps) # 数值稳定性条件

np.random.seed(127) 
phi  = np.random.rand(nx+2, ny+2) # [0,1)内的随机数
dphidt  = np.zeros((nx+2,ny+2)) 

def pbc(f):
    # 4 lines
    f[0,1:-1]=f[-2,1:-1]
    f[-1,1:-1]=f[1,1:-1]
    f[1:-1,0]=f[1:-1,-2]
    f[1:-1,-1]=f[1:-1,1]
    # 4 corners
    f[0,0]=f[-2,-2]
    f[0,-1]=f[-2,1]
    f[-1,0]=f[1,-2]
    f[-1,-1]=f[1,1]   

def calc_dphidt():
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            laplace = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1] - 4.0*phi[i,j])/dx/dx
            dgdphi = 2*phi[i,j]*(1-phi[i,j])*(1-2*phi[i,j])
            dphidt[i,j] = M_phi * (eps*eps*laplace - H*dgdphi)   

def update_phi():                
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            phi[i,j] += dphidt[i,j]*dt

# **********main function**********
nsteps = 200 # total number of time step
fig = plt.figure()
for step in range(1, nsteps+1):
    if step % 100 == 0 or step == 1 :
        print('nstep = ', step)
        plt.imshow(phi)
        plt.show()
    pbc(phi)
    calc_dphidt()
    update_phi()


      


