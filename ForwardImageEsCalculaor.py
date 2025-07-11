import numpy as np
from utility import util_cgfft
from utility import util_functions
from utility import setup_functions
from utility import generate_shapes 
import matplotlib.pyplot as plt
# Frequency = 400MHz
f = 4e8
wavelength = 3e8/f

# k := Wavenumber
k = 2*np.pi/wavelength 

# d := Dimension of imaging domain D in meters
# d = 2 
h = 0.025
d = 128*h

# R := Radius of measurement domain
R = 4

# M := Number of Receivers per illumination
M = 64

# V := Number of illuminations
V = 1

# Positions of Receivers and Transceivers. Shape = [M,2] and [V,2] respectively
# pos_S[i,:] = [x coordinate, y coordinate] of receiver #i 
pos_S = setup_functions.gen_pos_S(R, M, d) 
# pos_Tx[i,:] = [x coordinate, y coordinate] of transceiver #i 
pos_Tx = setup_functions.gen_pos_Tx(R*1.5, V, d)

############################################################# 
# Generating parameters for Forward at 56 X 56 Resolution
#############################################################
# The imaging domain is discretized into L X L cells
# For forward solver
L_forward = 32
n_forward = 0.035

pos_D_forward = setup_functions.gen_pos_D(d,L_forward,n_forward)
# print("position of grid points in forward solver: ", pos_D_forward)
# pos_D_forwardx= pos_D_forward[:,0].reshape(L_forward, L_forward)
# pos_D_forwardy= pos_D_forward[:,1].reshape(L_forward, L_forward)

# # Check for negative values in pos_D_forwardx and pos_D_forwardy
# has_neg_x = np.any(pos_D_forwardx < 0)
# has_neg_y = np.any(pos_D_forwardy < 0)
# print("pos_D_forwardx contains negative values:", has_neg_x)
# print("pos_D_forwardy contains negative values:", has_neg_y)
print("calculating e_forward...")
e_forward = setup_functions.gen_e(k, pos_D_forward, pos_Tx)
print("e_forward calculated over.")
# FFT representation of G_D matrix
print("constructing g_D_forward, g_D_fft_forward, g_D_fft_conj_forward...")
g_D_forward, g_D_fft_forward, g_D_fft_conj_forward = util_cgfft.construct_g_D(pos_D_forward, k, n_forward)
print("g_D_forward, g_D_fft_forward, g_D_fft_conj_forward constructed over.")
# G_S matrix for forward solver
print("constructing G_S_forward...")
# G_S_forward = util_functions.construct_G_S(pos_D_forward, pos_S, k, n_forward)
# Save G_S_forward to disk as a numpy file
# np.save('G_S_forward.npy', G_S_forward)
G_S_forward = np.load("G_S_forward.npy")
print("G_S_forward constructed over.")

# ############################################################# 
# # Creating the Image
# ############################################################# 
# max_contrast = 2.0
# x_au = generate_shapes.austria_multicontrast(L_forward,max_contrast,max_contrast,max_contrast)

# # Reshape profile into [N,1] vector
# x = np.reshape(x_au,[L_forward*L_forward,1])
# Create a 64x64 circular object with real value 2 and imaginary 0
L_circ = 32
radius = L_circ // 4  # radius of the circle
center = (L_circ // 2, L_circ // 2)

Y, X = np.ogrid[:L_circ, :L_circ]
dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
mask = dist_from_center <= radius

x_circ = np.zeros((L_circ, L_circ), dtype=np.complex128)
x_circ[mask] = 2.0 - 0.3j

# Reshape profile into [N,1] vector
x = np.reshape(x_circ, [L_circ * L_circ, 1])

############################################################# 
# Generating scattered field from profile
############################################################# 
# Run the forward solver
print('Running Forward Solver for %d illuminations'%(V))
print('This might take a while...')
y, _ = util_cgfft.cg_fft_forward_problem(x, G_S_forward, g_D_fft_forward, e_forward, 1e-9, e_forward, 10000)
# Add 25dB Gaussian Noise
print('Completed running the solver.')

print("y shape: ", y.shape)


# Reshape y into 128x128 for plotting
# Reshape y into 64x64 for plotting
y_reshaped_64 = np.reshape(y, (64, 64))
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.real(y_reshaped_64), cmap='rainbow', extent=[0, 64, 0, 64])
plt.title('Real part of scattered E-field (64x64)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Real(E)')

plt.subplot(1, 2, 2)
plt.imshow(np.imag(y_reshaped_64), cmap='rainbow', extent=[0, 64, 0, 64])
plt.title('Imaginary part of scattered E-field (64x64)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Imag(E)')

plt.tight_layout()
plt.savefig("y_64x64_plot.png")
plt.show()
# ############################################################# 
# # # Generating parameters for Inversion at 16 X 16 Resolution
# # #############################################################
# # L1 = 16
# # n_L1 = d/L1 # why  is this divide , what is n_L1


# # ## Ignore the warning which comes after running the code
# # max_contrast = 4.0
# # x_au = generate_shapes.austria_multicontrast(L_forward,max_contrast,max_contrast,max_contrast)

# # # Display Austria Profile
# # plt.imshow(np.real(x_au))
# # plt.xticks([L_forward*0.25, L_forward*0.5, L_forward*0.75], [-0.5, 0, 0.5],fontsize = '16')
# # plt.yticks([L_forward*0.25, L_forward*0.5, L_forward*0.75], [-0.5, 0, 0.5],fontsize = '16')
# # plt.xlabel('x (in m)', fontsize='16')
# # plt.ylabel('y (in m)', fontsize='16')
# # plt.title('Austria Profile', fontsize='18')
# # plt.colorbar()



