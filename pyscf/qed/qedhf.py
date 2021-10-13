#!/usr/bin/env python

import numpy as np
from pyscf import gto, scf
from itertools import product

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
E_threshold = 1e-8
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
