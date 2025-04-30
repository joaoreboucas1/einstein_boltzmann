import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, ifftn

# Grid size and resolution
N = 256  # Number of grid points along each dimension
box_size = 100.0  # Physical size of the box in Mpc

# Generate k-space grid
kx = np.fft.fftfreq(N, box_size/N)
ky = np.fft.fftfreq(N, box_size/N)
kz = np.fft.fftfreq(N, box_size/N)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
k = np.sqrt(KX**2 + KY**2 + KZ**2)  # Wavenumber magnitude

# Define power spectrum (e.g., P(k) ~ k^-3 for a simplistic cosmological case)
# P(k) = \Delta(k) / k^3
# Delta(k) = A_s * (k/k_p)^(n_s-1)
As = 2.1e-9
ns = 0.96
k_p = 0.05
Delta_k = As*(k/k_p)**(ns - 1)
P_k = np.where(k > 0, 2*np.pi**2*Delta_k/k**3, 0)

# Generate random complex field in Fourier space
rng = np.random.default_rng(seed=2611)
random_phase = rng.normal(size=(N, N, N)) + 1j * np.random.normal(size=(N, N, N))
delta_k = np.sqrt(P_k) * random_phase  # Multiply by sqrt(P(k)) to impose desired structure

# Transform back to real space
density_field = np.real(ifftn(delta_k))

# Visualizing a slice
plt.figure(figsize=(8, 6))
plt.imshow(density_field[:, :, N//2], cmap="inferno")
plt.colorbar(label="Density Contrast")
plt.title("2D Slice of a 3D Gaussian Matter Distribution")
plt.show()