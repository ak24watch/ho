import numpy as np
import scipy.special as sc

u0 = 1.2566370614e-6 	# Permeability
E0 =  8.85418782e-12; 	# Permitivity 
c = 3e8

# def gen_pos_D(d, L, n):
# 	# d : size of imaging domain, (assumed to be a square), in other context d_v is the internal field for the vth illumination
# 	# L : resolution of the imaging domain
# 	N = L*L # Number of grids

# 	pos_D = np.zeros((N,2))
# 	for i in range(N):
# 		pos_D[i,0] = np.mod(i,L)*n + n*0.5
# 		pos_D[i,1] = d - (np.floor(np.float32(i)/L)*n + n*0.5)
# 	return pos_D
def gen_pos_D(d, L, n):
    # d : size of imaging domain, (assumed to be a square), in other context d_v is the internal field for the vth illumination
    # L : resolution of the imaging domain
	h = 0.035
	x = y = np.linspace(-16*h, 16*h, 32)
	x, y = np.meshgrid(x, y, indexing="xy")
	pos_D = np.stack((x.ravel(), y.ravel()), axis=-1)
	return pos_D


# def gen_pos_S(R, M, d):
# 	# M : Number of sensors
# 	# R : Radius of measurement domain
# 	pos_S = np.zeros((M,2))
# 	for m in range(M):
# 		pos_S[m,0] = d*0.5 + R*np.cos(2*np.pi*((m+1)/M))
# 		pos_S[m,1] = d*0.5 + R*np.sin(2*np.pi*((m+1)/M))
# 	return pos_S

# def gen_pos_Tx(R, V, d):
# 	# pos_Tx[v,0], pos_Tx[v,1], pos_Tx[v,2]: x, y cooridinate and direction of propagation of wave for the vth illumination
# 	pos_Tx = np.zeros((V,3))
# 	for v in range(V):
# 		pos_Tx[v,0] = d*0.5 + R*np.cos(2*np.pi*(v/V))
# 		pos_Tx[v,1] = d*0.5 + R*np.sin(2*np.pi*(v/V))
# 		pos_Tx[v,2] = np.pi*2*v/V
# 	return pos_Tx
def gen_pos_Tx(R, V, d):
    # pos_Tx[v,0], pos_Tx[v,1], pos_Tx[v,2]: x, y cooridinate and direction of propagation of wave for the vth illumination
    pos_Tx = np.zeros((V, 3))
    for v in range(V):
        pos_Tx[v, 0] =  R * np.cos(2 * np.pi * (v / V))
        pos_Tx[v, 1] =  R * np.sin(2 * np.pi * (v / V))
        pos_Tx[v, 2] = np.pi * 2 * v / V
    return pos_Tx
# def gen_pos_S(n, d, u):
# 	x, y  = np.linspace(-64, 64, 128), np.linspace(-64, 64, 128)
# 	x, y = np.meshgrid(x, y, indexing='xy')
# 	pos_S = np.stack((x.ravel(), y.ravel()), axis=-1)
# 	return pos_S


def gen_pos_S(qq,qe,qt):
    # Create the full 128x128 grid from -64 to 64
 
    d = 128
    # u = 64
    h = 0.035
    x = y = np.linspace(-32*h, 32*h, 64)
    x, y = np.meshgrid(x, y, indexing="xy")

    # Reshape to (128*128, 2)
    pos_S = np.stack((x.ravel(), y.ravel()), axis=-1)

    # # Define mask to remove the center 64x64 block
    # mask_x = (x.ravel() < -u / 2) | (x.ravel() > u / 2)
    # mask_y = (y.ravel() < -u / 2) | (y.ravel() > u / 2)
    # mask = mask_x | mask_y  # Keep if outside in at least one axis

    # # Apply mask
    # pos_S = pos_S[mask]

    return pos_S
	

def gen_e(k, pos_D, pos_Tx):
	N = np.shape(pos_D)[0]
	V = np.shape(pos_Tx)[0]
	e = np.zeros((N,V), dtype = np.complex128)
	for v in range(V):
		theta = pos_Tx[v,2]
		for i in range(N):
			e[i,v] = np.exp( -1j*k*(pos_D[i,0]*np.cos(theta) + pos_D[i,1]*np.sin(theta) -  pos_Tx[v,0]*np.cos(theta) - pos_Tx[v,1]*np.sin(theta)) )
	return e

def gen_e_phase(k, pos_D, pos_Tx):
	N = np.shape(pos_D)[0]
	V = np.shape(pos_Tx)[0]
	e = np.zeros((N,V), dtype = np.float32)
	for v in range(V):
		theta = pos_Tx[v,2]
		for i in range(N):
			e[i,v] = k*(pos_D[i,0]*np.cos(theta) + pos_D[i,1]*np.sin(theta) -  pos_Tx[v,0]*np.cos(theta) - pos_Tx[v,1]*np.sin(theta)) 
	return e