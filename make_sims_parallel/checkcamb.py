import numpy as np
import matplotlib.pyplot as plt
import lenspyx
import sys, os
from lenspyx.utils import camb_clfile


#Read in power spectra. Make dimensionless.
cls_path = os.path.join(os.path.dirname(os.path.abspath(lenspyx.__file__)), 'data', 'cls')
cl_unl = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
lmax = 4096
dlmax = 1024
ell = np.arange(lmax+dlmax)
dimless_phi_cl = cl_unl['pp']

ls, cl_unl_camb, cl_len_camb, phi_camb = np.loadtxt('camb_lencl_phi.txt'))

plt.loglog(ls, cl_unl_camb, label = 'camb cl_unl')
plt.loglog(ell, cl_unl['tt'][0:lmax+dlmax], label = 'lenspyx cl_unl')
plt.loglog(ls, phi_camb, label = 'phi_camb')
plt.loglog(ell, dimless_phi_cl[0:lmax+dlmax], label = 'lenspyx phi')
plt.legend()
plt.savefig('compare_camb_lenspyx.pdf')

