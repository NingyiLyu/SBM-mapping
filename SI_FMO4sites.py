#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:51:31 2023

@author: ningyi
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import process_fidelity
from qiskit import Aer, execute  
from qiskit.extensions import RXGate, XGate, CXGate
from qutip import *
import scipy.linalg as LA
import matplotlib.pyplot as plt
import math
# Truncation of occupation number (maximally allowed excitation).
# In principle, could be set to any integer > N.
occ = 4

# Size of the N*N Hamiltonian.
N = 4   

# Imaginary unit.
EYE = 1j

# FMO electronic Hamiltonian.
# Matrix of the first 4 sites.
j_matrix_cm = np.array([[310.0, -97.9, 5.5, -5.8, 6.7, -12.1, -10.3, 37.5, ],
                        [-97.9, 230.0, 30.1, 7.3, 2.0, 11.5, 4.8, 7.9, ],
                        [5.5, 30.1, 0.0, -58.8, -1.5, -9.6, 4.7, 1.5, ],
                        [-5.8, 7.3, -58.8, 180.0, -64.9, -17.4, -64.4, -1.7, ],
                        [6.7, 2.0, -1.5, -64.9, 405.0, 89.0, -6.4, 4.5, ],
                        [-12.1, 11.5, -9.6, -17.4, 89.0, 320.0, 31.7, -9.7, ],
                        [-10.3, 4.8, 4.7, -64.4, -6.4, 31.7, 270.0, -11.4, ],
                        [37.5, 7.9, 1.5, -1.7, 4.5, -9.7, -11.4, 505.0, ]])

# Conversion to atomic unit for the matrix of the first 4 sites. 
# Corresponding to step 1 of figure 4. 
cm2au = 4.5563353e-6  # Conversion factor of wavenumbers to atomic units.
epsgam = np.zeros((N, N), dtype=complex)
for i in range(N):
    for j in range(N):
        epsgam[i][j] = j_matrix_cm[i][j] * cm2au

# Timestep of propagation.
tau = 200. 

# Number of timesteps.
nsc = 250

Uni_FMO_4states=LA.expm(-EYE*tau*epsgam) #numerically exact propagator for reference

#Creation of quantum circuit and transpilation. Corresponding to step 2 of figure 4.
q =  QuantumRegister(2,"qreg") #4 site model as 2-qubit circuit
qc = QuantumCircuit(q)
customUnitary = Operator(Uni_FMO_4states)
qc.unitary(customUnitary, [q[0], q[1]], label='custom')
newCircuit = transpile(qc, basis_gates=['u3','cz']) #transpilation

#routine for obtaining the matrix of a 1-qubit rotation from its Euler angles
def U3(theta,phi,lam):
    out = np.array([[np.cos(theta / 2),-np.exp(EYE * lam) * np.sin(theta / 2)],[np.exp(EYE * phi) * np.sin(theta / 2),np.exp(EYE * (phi + lam)) * np.cos(theta / 2)]])
    return out

#routine for extracting the gates out of newCircuit
orig_gate_array = []
for i in range(int(np.shape(newCircuit)[0])):
    if newCircuit[i][0].name == 'cz':
        orig_gate_array.append([np.diag([1,1,1,-1]),2])
    elif newCircuit[i][0].name == 'u3':
        if newCircuit[i][1][0].index == 0:
            orig_gate_array.append((U3(newCircuit[i][0].params[0],newCircuit[i][0].params[1],newCircuit[i][0].params[2]),0))
        elif newCircuit[i][1][0].index == 1:
            orig_gate_array.append((U3(newCircuit[i][0].params[0],newCircuit[i][0].params[1],newCircuit[i][0].params[2]),1))
        else:
            print('something is wrong')

#routine for creating the SNAIL gate from the 1-qubit rotation. Corresponding to steps 3 and 4 of Figure 4.             
def sbm_U(gate):
    gate = LA.logm(gate)*EYE #matrix logarithm for the effective Hamiltonian
    H11 = gate[0,0]
    H12 = gate[0,1]
    H21 = gate[1,0]
    H22 = gate[1,1]
    R12 = np.abs(H12)
    angle = np.exp(EYE*np.angle(H12))
    om = gate[1,1] - gate[0,0]
    g3 = - R12/3
    occ = N
    crea = create(occ) * np.conjugate(angle)
    anni = destroy(occ) * angle
    numoc = crea * anni
    out = H11 * np.eye(occ) + 2. * R12 * (crea+anni) + om * numoc + g3 * (crea+anni) ** 3 - g3 * (crea ** 3 + anni ** 3) #SNAIL programming
    out = LA.expm( -EYE * np.array(out))
    out_trun = np.zeros((2,2),dtype=complex)
    for i in range(2):
        for j in range(2):
            out_trun[i,j] = out[i,j]
    return out_trun,om,g3,H11,R12

#Preparation of bosonic CZ gate, as described by Appendix D.
def CZ(x):
    anni_a = destroy(x)
    crea_a = create(x)
    out = LA.expm(-EYE * np.pi * np.kron(crea_a * anni_a,crea_a * anni_a))
    return out

#Assembling all prepared bosonic gates. Corresponding to step 5 of Figure 4. 
gate_array = np.array(orig_gate_array)
for i in range(np.shape(gate_array)[0]):
    if gate_array[i][1] == 2:
        gate_array[i][0] = CZ(2)
    else:
        gate_array[i][0] = sbm_U(orig_gate_array[i][0])

#Run numerical simulation
Op = np.eye(2*2,dtype=complex)
for i in range(np.shape(gate_array)[0]):
    if gate_array[i][1] == 2:
        Op = gate_array[i][0] @ Op
        #print(i)
    elif gate_array[i][1] == 1:
        if gate_array[i-1][1] == 0:
            Op = np.kron(gate_array[i][0][0],gate_array[i-1][0][0]) @ Op
        elif gate_array[i+1][1] == 0:
            Op = np.kron(gate_array[i][0][0],gate_array[i+1][0][0]) @ Op
        else:
            Op = np.kron(gate_array[i][0][0].np.eye(2)) @ Op
    else:
        Op = Op
sbm_prop = Op

#Initialize at vacuum state
psivec_ref = np.zeros((4),dtype=complex)
psivec_ref[0] = 1.
psivec_sbm = np.zeros((4),dtype=complex)
psivec_sbm[0] = 1.
ps1_ref = np.zeros((nsc),dtype=complex)
ps2_ref = np.zeros((nsc),dtype=complex)
ps1_sbm = np.zeros((nsc),dtype=complex)
ps2_sbm = np.zeros((nsc),dtype=complex)
t = np.arange(0,nsc * tau,tau)

#Time propagation
for i in range(nsc):
    psivec_ref = Uni_FMO_4states @ psivec_ref
    ps1_ref[i] = np.conj(psivec_ref[0]) * psivec_ref[0]
    ps2_ref[i] = np.conj(psivec_ref[1]) * psivec_ref[1]
    psivec_sbm = sbm_prop @ psivec_sbm
    ps1_sbm[i] = np.conj(psivec_sbm[0]) * psivec_sbm[0]
    ps2_sbm[i] = np.conj(psivec_sbm[1]) * psivec_sbm[1]

#Plot results to get Figure 5b. 
plt.figure(dpi=600)
plt.xlabel('time(a.u.)')
plt.ylabel('Population')
plt.xlim(0,50000)
#plt.ylim(0,1.)
plt.plot(t,np.abs(ps1_ref),'r',label='reference')
plt.scatter(t[1::5],np.abs(ps1_sbm)[1::5],c='b',label='sbm')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))