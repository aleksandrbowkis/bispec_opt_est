import numpy as np
import matplotlib.pyplot as plt

L_nonoise, snr_nonoise = np.loadtxt('snr_nonoise_rlmax3000.txt', unpack=True)
L_noise, snr_noise = np.loadtxt('snr_noise_rlmax3000.txt', unpack=True)

plt.plot(L_nonoise, snr_nonoise, label = 'snr for temp only optimal estimator of lensing kappa bispectrum no noise, rlmax = 3000')
plt.plot(L_noise, snr_noise, label = 'snr temp only opt est kappa bispec with noise, rlmax = 3000')
plt.legend()
plt.show()
plt.savefig('temp_snr.pdf')
