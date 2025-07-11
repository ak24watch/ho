import numpy as np

import scipy.special as sc
import numpy.linalg as linalg



# 

# Construct G_S matrix constant for all iterations
def construct_G_S(pos_D, pos_S, k, n):
	counter = 0
	M = np.shape(pos_S)[0]
	N = np.shape(pos_D)[0]
	G_S = np.zeros([M,N],dtype =np.complex128)	
	a = (n**2/np.pi)**0.5

	for i in range(M):
		for j in range(N):
			counter += 1
			rho_ij = linalg.norm(pos_S[i,:]-pos_D[j,:],2)
			G_S[i,j] = -1j*0.5*np.pi*k*a*sc.j1(k*a)*sc.hankel2(0,k*rho_ij)
			print("counter =", counter, " out of ", M*N, end='\r')
	return G_S

# scattering fields add noise
def add_noise(signal, SNR):
	signal_shape = np.shape(signal)
	signal_power = np.linalg.norm(signal,'fro')**2
	sigma = ((10**(-SNR/10))*signal_power/np.prod(signal_shape))**0.5
	noise = sigma*np.random.randn(*signal_shape)

	return signal + noise


# calcuate error last after model output check with ground truth
def shape_error(contrast, contrast_true):
	N = np.shape(contrast)[0]
	diff = np.divide(np.abs(contrast - contrast_true),np.abs(contrast_true + 1))
	err_total = np.sum(diff)/N
	err_internal = np.sum(diff*(abs(contrast_true)> 1e-3))/np.sum(np.asarray(abs(contrast_true)>1e-3,dtype=np.float32))
	return [err_internal, err_total]


