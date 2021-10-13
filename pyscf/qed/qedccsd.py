#!/usr/bin/env python

import numpy as np
from pyscf import gto, scf
from itertools import product
from numba import jit
import math
import time
from amplitude_equations import *

mol = gto.Mole()
mol.unit = 'A'
mol.atom = '''
O            0.000000000000     0.000000000000    -0.068516219320
H            0.000000000000    -0.790689573744     0.543701060715
H            0.000000000000     0.790689573744     0.543701060715
'''
mol.basis = 'cc-pvdz'
mol.verbose = 7
mol.build()

mymf = scf.RHF(mol)
mymf.kernel()

divider = f"{''.join(['*']*8)} Running QED-HF {''.join(['*']*8)}"
print(f'\n{divider}\n')

au2ev = 27.211386245988
# cavity parameters
n_photon_states = 2
# cavity frequency
cavity_frequency = 2.  # eV
cavity_frequency /= au2ev
# coupling strength
lambda_cav = np.array([0.0, 0.0, 0.05])
lambda_x = lambda_cav[0]
lambda_y = lambda_cav[1]
lambda_z = lambda_cav[2]

#
# SCF initialization
#
S = mymf.get_ovlp()

# Build core Hamiltonian
H_core = mymf.get_hcore()

# nuclear-neculear repulsion energy
E_nuc = mol.energy_nuc()

# guess density matrix
D = mymf.make_rdm1()

diff_SCF_E = 1e10
E_old = 0
E_threshold = 1e-16
MAXITER = 50000

#
# photonic initialization
#

# nuclear dipole
charges = mol.atom_charges()
r = mol.atom_coords()  # Assume origin is at (0,0,0)
nuc_dip = np.einsum('g,gx->x', charges, r)
nuc_dip_x = nuc_dip[0]
nuc_dip_y = nuc_dip[1]
nuc_dip_z = nuc_dip[2]

# electronic dipole
nao = mol.nao
dip_ao = -1. * mol.intor('int1e_r').reshape(3,nao,nao)
dipole0 = dip_ao[0]
dipole1 = dip_ao[1]
dipole2 = dip_ao[2]

# electronic quadrupole
quadrupole = -1. * mol.intor('int1e_rr').reshape(3,3,nao,nao)
quadrupole0 = quadrupole[0, 0]
quadrupole1 = quadrupole[0, 1]
quadrupole2 = quadrupole[0, 2]
quadrupole3 = quadrupole[1, 1]
quadrupole4 = quadrupole[1, 2]
quadrupole5 = quadrupole[2, 2]

quadrupole = np.zeros(6)
quadrupole_0 = 0.5 * lambda_x * lambda_x * quadrupole0
quadrupole_1 = 1.0 * lambda_x * lambda_y * quadrupole1
quadrupole_2 = 1.0 * lambda_x * lambda_z * quadrupole2
quadrupole_3 = 0.5 * lambda_y * lambda_y * quadrupole3
quadrupole_4 = 1.0 * lambda_y * lambda_z * quadrupole4
quadrupole_5 = 0.5 * lambda_z * lambda_z * quadrupole5

quadrupole_scaled_sum = (quadrupole_0 + quadrupole_1 + quadrupole_2 +
                         quadrupole_3 + quadrupole_4 + quadrupole_5)

# for two-electron part of 1/2 (lambda.d)^2 (1/2 picked up in rhf.cc)
dipole_scaled_sum = (lambda_x * dipole0 +
                     lambda_y * dipole1 +
                     lambda_z * dipole2)

# e-(n-<d>) contribution 0.5 * 2 (lambda . de) ( lambda . (dn - <d>) )
tot_dip_x = nuc_dip_x
tot_dip_y = nuc_dip_y
tot_dip_z = nuc_dip_z

# e contribution: lambda . de
el_dipdot = (lambda_x * dipole0 +
             lambda_y * dipole1 +
             lambda_z * dipole2)

# n-<d> contribution: lambda . (dn - <d>)
nuc_dipdot = 0
nuc_dipdot += lambda_x * (nuc_dip_x - tot_dip_x)
nuc_dipdot += lambda_y * (nuc_dip_y - tot_dip_y)
nuc_dipdot += lambda_z * (nuc_dip_z - tot_dip_z)

# e-(n-<d>) contribution 0.5 * 2 (lambda . de) ( lambda . (dn - <d>) )
scaled_e_n_dipole_squared = el_dipdot * nuc_dipdot

# constant terms:  0.5 ( lambda . ( dn - <d> ) )^2 = 0.5 ( lambda . <de> )^2
average_electric_dipole_self_energy = 0.5 * nuc_dipdot * nuc_dipdot

# Begin Iterations
for scf_iter in range(1, MAXITER+1):
    J, K = mymf.get_jk(dm=D)

    D /= 2.
    J /= 2.
    K /= 2.
    
    F = H_core + 2 * J - K
    # F = H_core + J - 0.5 * K 

    oei = np.zeros((H_core.shape[0], H_core.shape[1]))
    if n_photon_states > 1:
        # update cavity terms
        nS = n_photon_states

        e_dip_x = np.einsum('pq, pq ->', 2 * D, dipole0, optimize=True)
        e_dip_y = np.einsum('pq, pq ->', 2 * D, dipole1, optimize=True)
        e_dip_z = np.einsum('pq, pq ->', 2 * D, dipole2, optimize=True)

        # evaluate the total dipole moment:
        #TODO confirm the sign of dipole
        tot_dip_x = e_dip_x + nuc_dip_x
        tot_dip_y = e_dip_y + nuc_dip_y
        tot_dip_z = e_dip_z + nuc_dip_z

        # e-(n-<d>) contribution 0.5 * 2 (lambda . de) ( lambda . (dn - <d>) )

        # e contribution: lambda . de
        el_dipdot = (lambda_x * dipole0 +
                     lambda_y * dipole1 +
                     lambda_z * dipole2)

        # n-<d> contribution: lambda . (dn - <d>)
        nuc_dipdot = 0
        nuc_dipdot += lambda_x * (nuc_dip_x - tot_dip_x)
        nuc_dipdot += lambda_y * (nuc_dip_y - tot_dip_y)
        nuc_dipdot += lambda_z * (nuc_dip_z - tot_dip_z)                   

        # e-(n-<d>) contribution 0.5 * 2 (lambda . de) ( lambda . (dn - <d>) )
        scaled_e_n_dipole_squared = el_dipdot * nuc_dipdot

        # constant terms:  0.5 ( lambda . ( dn - <d> ) )^2 = 0.5 ( lambda . <de> )^2
        average_electric_dipole_self_energy = 0.5 * nuc_dipdot * nuc_dipdot

        # dipole self energy:

        # e-n term
        oei = scaled_e_n_dipole_squared

        # one-electron part of e-e term
        oei -= quadrupole_scaled_sum

        # two-electron part of e-e term (J)
        scaled_mu = np.einsum('pq,pq->', D, dipole_scaled_sum, optimize=True)
        F += 2 * scaled_mu * dipole_scaled_sum
        #K
        F -= np.einsum('pr,qs,rs->pq', dipole_scaled_sum, dipole_scaled_sum, D, optimize=True)
        F += oei
    
    E_new = (np.einsum('pq,qp->', (oei + H_core + F), D, optimize=True) + E_nuc + average_electric_dipole_self_energy)

    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E' % (scf_iter, E_new, E_new - E_old))

    # SCF Converged?
    if (abs(E_new - E_old) < E_threshold):
        break
    E_old = E_new

    mo_energy, mo_coeff = mymf.eig(F, S)
    mo_occ = mymf.get_occ(mo_energy, mo_coeff)
    D = mymf.make_rdm1(mo_coeff, mo_occ)


    if (scf_iter == MAXITER):
        raise Exception("Maximum number of SCF iterations exceeded.")

# Post iterations
print('\nSCF finished.')
print('Final RHF Energy: %.14f [Eh]' % (E_new))


def ao_to_mo_transform_full(h_core, g_ao, C):
    tmp = np.einsum('pi,pqrs->iqrs', C, g_ao, optimize=True)
    tmp = np.einsum('iqrs,rj->iqjs', tmp, C, optimize=True)
    tmp = np.einsum('qa,iqjs->iajs', C, tmp, optimize=True)
    g_mo = np.einsum('iajs,sb->iajb', tmp, C, optimize=True)

    h_mo = np.einsum('pi,pq,qj -> ij', C, h_core, C, optimize=True)

    return h_mo, g_mo


g_ao = mol.intor('int2e_sph')

lx2 = lambda_x * lambda_x
ly2 = lambda_y * lambda_y
lz2 = lambda_z * lambda_z
dipole_self_energy = (lx2 * np.einsum('pq,rs->pqrs', dipole0, dipole0)
                      + ly2 * np.einsum('pq,rs->pqrs', dipole1, dipole1)
                      + lz2 * np.einsum('pq,rs->pqrs', dipole2, dipole2))
if n_photon_states > 1:
    g_ao += dipole_self_energy

# ==> core Hamiltoniam <==
h = H_core
if n_photon_states > 1:
    h += oei

# C = mymf.mo_coeff
C = mo_coeff
print(f"MO coeff: {mo_coeff}")
h_mo_sf, g_mo_sf = ao_to_mo_transform_full(h, g_ao, C)

@jit(nopython=True)
def sf_to_so(g_mo_sf, nso):
    g_mo = np.zeros((nso, nso, nso, nso))
    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    value1 = g_mo_sf[p// 2, q// 2, r// 2, s// 2] * (p % 2 == q % 2) * (r % 2 == s % 2)
                    value2 = g_mo_sf[p// 2, s// 2, r// 2, q// 2] * (p % 2 == s % 2) * (q % 2 == r % 2)
                    g_mo[p,q,r,s] = value1 - value2
    return g_mo

nmos_sf = g_mo_sf.shape
assert(np.all([nmos_sf[x] == nmos_sf[0] for x in [1,2,3]]))
nmo_sf = nmos_sf[0]
nso = 2 * nmo_sf
g_mo = sf_to_so(g_mo_sf, nso)

g_mo = g_mo.transpose(0, 2, 1, 3)

#defined as g_{pq}^{X} in White's paper
coupling_factor_x = lambda_x*math.sqrt(cavity_frequency/2)
coupling_factor_y = lambda_y*math.sqrt(cavity_frequency/2)
coupling_factor_z = lambda_z*math.sqrt(cavity_frequency/2)

dipole_x = np.kron(np.eye(2), dipole0)
dipole_y = np.kron(np.eye(2), dipole1)
dipole_z = np.kron(np.eye(2), dipole2)

dipole_x_tr = np.einsum('pi,pq,qj -> ij', C, dipole0, C, optimize=True)
dipole_y_tr = np.einsum('pi,pq,qj -> ij', C, dipole1, C, optimize=True)
dipole_z_tr = np.einsum('pi,pq,qj -> ij', C, dipole2, C, optimize=True)
#- sign is in the function call
dp_sf = (dipole_x_tr * coupling_factor_x
    + dipole_y_tr * coupling_factor_y
    + dipole_z_tr * coupling_factor_z)

G = 0

print(f"F in AO: {F}\n")
f_tmp = F.dot(C)
f = C.transpose().dot(f_tmp)
print(f"f in MO: {f}\n")

f_mo = np.zeros((nso,nso))
dp = np.zeros((nso,nso))

e = mymf.mo_energy
eps = np.diag(e)
for p in range(nso):
    for q in range(nso):
        f_mo[p, q] = f[p // 2, q//2] * (p % 2 == q % 2)
        dp[p, q] = dp_sf[p // 2, q // 2] * (p % 2 == q % 2)

nocc = int(np.sum(mymf.mo_occ))
nvir = 2 * nao - nocc
#pure singles amplitudes electron/photon
t1_10 = np.zeros((nvir, nocc))
t1_01 = 0
#pure double amplitudes electron/photon
t2_20 = np.zeros((nvir, nvir, nocc, nocc))
t2_02 = 0
#mixed double amplitudes electron-photon
t2_11 = np.zeros((nvir, nocc))
t2_21 = np.zeros((nvir, nvir, nocc, nocc))
t2_12 = np.zeros((nvir, nocc))
t2_22 = np.zeros((nvir, nvir, nocc, nocc))

# MP2
eps = f_mo.diagonal()
print(f"epsilon: {eps}")
eps_occ = eps[:nocc]
eps_vir = eps[nocc:]
eps_vir_p_w = eps[nocc:] + cavity_frequency
e_denom_p = 1 / (eps_occ.reshape(-1, 1) - eps_vir_p_w)
t2_11 = np.einsum('ai,ia -> ai', -dp[nocc:,:nocc], e_denom_p, optimize=True)

e_denom_e = 1 / (eps_occ.reshape(-1, 1, 1, 1) + eps_occ.reshape(-1, 1) - eps_vir.reshape(-1, 1, 1) - eps_vir)
t2_20 = np.einsum('iajb,abij->abij', e_denom_e, g_mo[nocc:,nocc:,:nocc,:nocc], optimize = True)

E_QED_MP2  = 0.25*np.einsum('abij,abij->', t2_20, g_mo[nocc:,nocc:,:nocc,:nocc], optimize = True)
print('E_QED_MP2            = %4.15f' % (E_QED_MP2))
E_QED_MP2 -= 1.0 * np.einsum('ia,ai->', dp[:nocc, nocc:], t2_11)
print('E_QED_MP2            = %4.15f' % (E_QED_MP2))
print('E_QED_MP2 tot        = %4.15f' % (E_QED_MP2+E_new))

E_CCSD_old = 0
tol = 1e-16
MAXITER = 5000
time_average = 0

### Setup DIIS
diis_vals_t1_10 = [t1_10.copy()]
diis_vals_t2_20 = [t2_20.copy()]

do_t1_01 = True
do_t2_11 = True
do_t2_21 = False
do_t2_02 = True
do_t2_12 = True
do_t2_22 = False
#conventional
if do_t1_01 == False and do_t2_11 == False and do_t2_21 == False and do_t2_02 == False and do_t2_12 == False and do_t2_22 == False:
    print("doing conventional CCSD")
#Deprince, White, Full doubles
if do_t1_01 == True and do_t2_11 == True and do_t2_21 == True and do_t2_02 == False and do_t2_12 == False and do_t2_22 == False:
    print("doing QED-CCSD-1 (or QED-CCSD-21)")
if do_t1_01 == True and do_t2_11 == True and do_t2_21 == False and do_t2_02 == True and do_t2_12 == True and do_t2_22 == False:
    print("doing QED-CCSD-White (or QED-CCSD-12)")
if do_t1_01 == True and do_t2_11 == True and do_t2_21 == True and do_t2_02 == True and do_t2_12 == True and do_t2_22 == True:
    print("doing QED-CCSD-Full (or QED-CCSD-22)")
#up to singles photonic excitations
if do_t2_11:
    diis_vals_t2_11 = [t2_11.copy()]
if do_t2_21:
    diis_vals_t2_21 = [t2_21.copy()]
#up to double photonic excitations
if do_t2_12:
    diis_vals_t2_12 = [t2_12.copy()]
if do_t2_22:
    diis_vals_t2_22 = [t2_22.copy()]

diis_errors = []
max_diis = 20

print('\nStarting with the CCSD calculation:\n')
print('Iter  Energy(CCSD)      E_diff    time(sec)')
for ccsd_iter in range(1, MAXITER + 1):

    t_start = time.time()

    # Save new amplitudes
    oldt1_10 = t1_10.copy()
    oldt2_20 = t2_20.copy()
    if do_t2_11:
        oldt2_11 = t2_11.copy()
    if do_t2_21:
        oldt2_21 = t2_21.copy()
    if do_t2_12:
        oldt2_12 = t2_12.copy()
    if do_t2_22:
        oldt2_22 = t2_22.copy()

    #singles
    t1_10 = ccsd_t1_10(f_mo, g_mo, dp, G, cavity_frequency, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
    if do_t1_01:
        t1_01 = ccsd_t1_01(f_mo, g_mo, dp, G, cavity_frequency, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
    #pure doubles
    t2_20 = ccsd_t2_20(f_mo, g_mo, dp, G, cavity_frequency, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
    if do_t2_02:
        t2_02 = ccsd_t2_02(f_mo, g_mo, dp, G, cavity_frequency, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
    #mixed doubles
    if do_t2_11:
        t2_11 = ccsd_t2_11(f_mo, g_mo, dp, G, cavity_frequency, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
    if do_t2_21:
        t2_21 = ccsd_t2_21(f_mo, g_mo, dp, G, cavity_frequency, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
    if do_t2_12:
        t2_12 = ccsd_t2_12(f_mo, g_mo, dp, G, cavity_frequency, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)
    if do_t2_22:
        t2_22 = ccsd_t2_22(f_mo, g_mo, dp, G, cavity_frequency, t1_10, t1_01, t2_20, t2_02, t2_11, t2_21, t2_12, t2_22)

    #E_CCSD_new += 1.0 * G * t1_01
    E_CCSD_new = 1.0 * np.einsum('ia,ai->', f_mo[:nocc, nocc:], t1_10)
    E_CCSD_new = 0.25 * np.einsum('ijab,baji->', g_mo[:nocc, :nocc, nocc:, nocc:], t2_20)
    E_CCSD_new -= 1.0 * np.einsum('ia,ai->', dp[:nocc, nocc:], t2_11)
    E_CCSD_new -= 1.0 * t1_01 * np.einsum('ia,ai->', dp[:nocc, nocc:], t1_10)
    E_CCSD_new += -0.5 * np.einsum('ijab,bi,aj->', g_mo[:nocc, :nocc, nocc:, nocc:], t1_10, t1_10)

    t_total = time.time() - t_start
    time_average += t_total
    print('%3d:  %4.10f  %1.5E   %.3f' % (ccsd_iter, E_CCSD_new, \
                                          abs(E_CCSD_new - E_CCSD_old), t_total))

    if (abs(E_CCSD_new - E_CCSD_old) < tol):
        break

    # Add DIIS vectors
    diis_vals_t1_10.append(t1_10.copy())
    diis_vals_t2_20.append(t2_20.copy())
    if do_t2_11:
        diis_vals_t2_11.append(t2_11.copy())
    if do_t2_21:
        diis_vals_t2_21.append(t2_21.copy())
    if do_t2_12:
        diis_vals_t2_12.append(t2_12.copy())
    if do_t2_22:
        diis_vals_t2_22.append(t2_22.copy())

    # Build new error vector
    error_t1_10 = (t1_10 - oldt1_10).ravel()
    error_t2_20 = (t2_20 - oldt2_20).ravel()
    if do_t2_11:
        error_t2_11 = (t2_11 - oldt2_11).ravel()
    if do_t2_21:
        error_t2_21 = (t2_21 - oldt2_21).ravel()
    if do_t2_12:
        error_t2_12 = (t2_12 - oldt2_12).ravel()
    if do_t2_22:
        error_t2_22 = (t2_22 - oldt2_22).ravel()

    #conventional
    if do_t1_01 == False and do_t2_11 == False and do_t2_21 == False and do_t2_02 == False and do_t2_12 == False and do_t2_22 == False:
        diis_errors.append(np.concatenate((error_t1_10, error_t2_20)))
    # Deprince, White, Full doubles
    if do_t1_01 == True and do_t2_11 == True and do_t2_21 == True and do_t2_02 == False and do_t2_12 == False and do_t2_22 == False:
        diis_errors.append(np.concatenate((error_t1_10, error_t2_20, error_t2_11, error_t2_21)))
    if do_t1_01 == True and do_t2_11 == True and do_t2_21 == False and do_t2_02 == True and do_t2_12 == True and do_t2_22 == False:
        diis_errors.append(np.concatenate((error_t1_10, error_t2_20, error_t2_11, error_t2_12)))
    if do_t1_01 == True and do_t2_11 == True and do_t2_21 == True and do_t2_02 == True and do_t2_12 == True and do_t2_22 == True:
        diis_errors.append(np.concatenate((error_t1_10, error_t2_20, error_t2_11, error_t2_21, error_t2_12, error_t2_22)))


    E_CCSD_old = E_CCSD_new


    if ccsd_iter >= 1:

        # Limit size of DIIS vector
        if (len(diis_vals_t1_10) > max_diis):
            del diis_vals_t1_10[0]
            del diis_vals_t2_20[0]
            if do_t2_11:
                del diis_vals_t2_11[0]
            if do_t2_21:
                del diis_vals_t2_21[0]
            if do_t2_12:
                del diis_vals_t2_12[0]
            if do_t2_22:
                del diis_vals_t2_22[0]
            del diis_errors[0]

        diis_size = len(diis_vals_t1_10) - 1

        # Build error matrix B, [Pulay:1980:393], Eqn. 6, LHS
        B = np.ones((diis_size + 1, diis_size + 1)) * -1
        B[-1, -1] = 0

        for n1, e1 in enumerate(diis_errors):
            for n2, e2 in enumerate(diis_errors):
                # Vectordot the error vectors
                B[n1, n2] = np.dot(e1, e2)

        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

        # Build residual vector, [Pulay:1980:393], Eqn. 6, RHS
        resid = np.zeros(diis_size + 1)
        resid[-1] = -1

        # Solve Pulay equations, [Pulay:1980:393], Eqn. 6
        ci = np.linalg.solve(B, resid)

        # Calculate new amplitudes
        t1_10[:] = 0
        t2_20[:] = 0
        if do_t2_11:
            t2_11[:] = 0
        if do_t2_21:
            t2_21[:] = 0
        if do_t2_12:
            t2_12[:] = 0
        if do_t2_22:
            t2_22[:] = 0
        for num in range(diis_size):
            t1_10 += ci[num] * diis_vals_t1_10[num + 1]
            t2_20 += ci[num] * diis_vals_t2_20[num + 1]
            if do_t2_11:
                t2_11 += ci[num] * diis_vals_t2_11[num + 1]
            if do_t2_21:
                t2_21 += ci[num] * diis_vals_t2_21[num + 1]
            if do_t2_12:
                t2_12 += ci[num] * diis_vals_t2_12[num + 1]
            if do_t2_22:
                t2_22 += ci[num] * diis_vals_t2_22[num + 1]
    # End DIIS amplitude update

#H2O (DePrince) QED-CCSD-1/cc-pVDZ omega=2eV lambda=(0.0,0.0,0.05), E_threshold = 1e-13, tol = 1e-10
E_QED_CCSD_1_2eV_lambda_05_H2O_DZ = -0.217101266675235
#or -0.217101278351680 due to different fock construction
#H2O (DePrince) QED-CCSD-full/cc-pVDZ omega=2eV lambda=(0.0,0.0,0.05), E_threshold = 1e-13, tol = 1e-10
E_QED_CCSD_full_2eV_lambda_05_H2O_DZ = -0.217102128591929
#H2O (DePrince) QED-CCSD-white/cc-pVDZ omega=2eV lambda=(0.0,0.0,0.05), E_threshold = 1e-13, tol = 1e-10
E_QED_CCSD_white_2eV_lambda_05_H2O_DZ = -0.217069141583762
print('E_QED_CCSD           = %4.15f' % (E_CCSD_new))
print('E_QED_CCSD tot       = %4.15f' % (E_CCSD_new+E_new))