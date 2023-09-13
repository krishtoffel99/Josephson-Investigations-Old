import kwant
import scipy
import numpy as np
import matplotlib.pyplot as plt 
from scipy import sparse
import scipy.sparse.linalg as sla
import tinyarray as tiny

from tqdm import tqdm

sx = tiny.array([[0 , 1] , [1 , 0]])
sy = tiny.array([[0 , -1j] , [1j , 0]])
sz = tiny.array([[1 , 0] , [0 , -1]])


I = tiny.array([[1 , 0] , [0 , 1]])
t_plus = 0.5*(np.kron(sx , I) + 1j*np.kron(sy , I))
t_minus = t_plus.T

# Josephson vortices:
def make_josephson_junction(**params):
    # This function makes the Josephson junction system for a dictionary of parameters passed to it.

    a = params['a'] # Lattice constant
    L = params['L'] # Length of the junction
    W = params['W'] # Width of system
    d = params['d'] # Width of the junction
    flux = params['flux'] # Number of flux quanta threaded through the system
    offset = params['offset'] # Josephson offset phase
    A = params['A'] # QAH kinetic parameter
    B = params['B'] # QAH momentum-dependent Zeeman coupling
    C = params['C'] # QAH Isotropic Zeeman coupling
    D = params['D'] # Bulk Superconducting gap
    u = params['u'] # Chemical potential


    # scaled parameters:
    l = int(L/a)
    w = int(W/a)
    d_scaled = int(d/a)

    lat = kwant.lattice.square(a, norbs = 4)
    syst = kwant.Builder(particle_hole = np.kron(sy , sy)) 
    
    m = (w - 1)/2
    # Phase profile due to magnetic field:
    def phase(i , j):
        if (m + d_scaled)<= j < w and 0<= i < l:
            return (2*np.pi)*(flux)*(1/l)*i + (2*np.pi*offset)
        else:
            return 0

    # Defining lattice hoppings:
    for i in range(l):
        for j in range(w):
            if 0 <= j <= (m-d_scaled):
                syst[lat(i , j)] = C*np.kron(I , sz) + 4*(B/a**2)*np.kron(I , sz)+ D*np.exp(1j*phase(i , j))*t_plus + D*np.exp(-1j*phase(i , j))*t_minus + u*np.kron(sz , I)

            if (m-d_scaled)< j < (m+d_scaled):
                syst[lat(i , j)] = C*np.kron(I , sz) + 4*(B/a**2)*np.kron(I , sz) + u*np.kron(sz , I)

            if (m+d_scaled)<= j < w:
                syst[lat(i , j)] = C*np.kron(I , sz) + 4*(B/a**2)*np.kron(I , sz)+ D*np.exp(1j*phase(i , j))*t_plus + D*np.exp(-1j*phase(i ,j))*t_minus + u*np.kron(sz , I)

            if j > 0:
                    syst[lat(i , j) , lat(i , j-1)] = -1j*(A/a)*np.kron(sz , sy) - (B/a**2)*np.kron(I , sz)

            if i > 0:
                syst[lat(i , j) , lat(i-1 , j)] = -1j*(A/a)*np.kron(sz , sx) - (B/a**2)*np.kron(I , sz)
    
    syst = syst.finalized()

    return syst

# These are various solving and plotting functions:
def solve_me(syst, number):
    hamiltonian = syst.hamiltonian_submatrix(sparse = True)

    eigenvalues , eigenvectors = sla.eigsh(hamiltonian , k = number , sigma = 0.0 , which = 'LM')

    idx = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[: , idx]

    return eigenvalues , eigenvectors

def generalised_solver(syst, n = 6, mode = 'sign'):
    """
     Solves Hamiltonian for system for first 'n' eigenvectors and eigenvalues. 
     Arguments:
     syst: kwant system for josephson junction .
     n: Number of eigenvectors and eigenvalues to obtain.
     mode: String that determines the ordwering convention for the eigenvectors and eigenvalues. Default set to 'sign'
           - 'sign' orders eigenvalues in terms of sign and ascending magnitude e.g. a spectrum consisting of eigenvalues [-3 , -2 , -1 , 1 , 2 , 3]
             is reordered to [-1, -2 , -3 , 1 , 2 , 3].
           - 'abs' arranges eigenvalues in order of ascending absolute value e.g. a spectrum consisting of eigenvalues [-3 , -2 , -1 , 1 , 2 , 3]
             is reordered to [-1 , 1 , -2 , 2 , -3 , 3]. 
     Returns:
     eigenvalues array, eigenvector array.

    """
    hamiltonian = syst.hamiltonian_submatrix(sparse = True)
    eigenvalues , eigenvectors = sla.eigsh(hamiltonian , k = n , sigma = 0.0 , which = 'LM')

    if mode == 'abs':
        idx = np.argsort(np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[: , idx]
        return eigenvalues , eigenvectors

    if mode == 'sign':
        eigenvalues_negative = eigenvalues[eigenvalues <= 0]
        eigenvalues_positive = eigenvalues[eigenvalues >= 0]

        eigenvalues_negative_sorted = eigenvalues_negative[np.argsort(np.abs(eigenvalues_negative))]
        eigenvalues_positive_sorted = eigenvalues_positive[np.argsort(np.abs(eigenvalues_positive))]
        # Gluing the two together
        eigenvalues_new =  np.concatenate((eigenvalues_negative_sorted, eigenvalues_positive_sorted) , axis = None)

        # Ordering eigenvectors:
        eigenvectors_negative = eigenvectors[: , eigenvalues <= 0]
        eigenvectors_positive = eigenvectors[: , eigenvalues >= 0]
        eigenvectors_negative_sorted = eigenvectors_negative[: , np.argsort(np.abs(eigenvalues_negative))]
        eigenvectors_positive_sorted = eigenvectors_positive[: , np.argsort(np.abs(eigenvalues_positive))]

        eigenvectors_new = np.concatenate((eigenvectors_negative_sorted , eigenvectors_positive_sorted) , axis = 1)

        return eigenvalues_new , eigenvectors_new


def see_the_funk(k , ev , params):
    psi = ev[: , k]
    innerprod = np.conjugate(psi) * psi
    prob_density = np.array([sum(innerprod[i : i + 4]) for i in range(0 , len(innerprod), 4)])
    wavefunc = np.reshape(prob_density , ((int(params['L']/params['a'])) , int(params['W']/params['a'])))
    return np.real(wavefunc.T)

def see_the_funk_2(ev , params):
    psi = ev
    innerprod = np.conjugate(psi) * psi
    prob_density = np.array([sum(innerprod[i : i + 4]) for i in range(0 , len(innerprod), 4)])
    wavefunc = np.reshape(prob_density , ((int(params['L']/params['a'])) , int(params['W']/params['a'])))
    return np.real(wavefunc.T)

def inner(a , b):
    return np.conjugate(a) @ b

def amplitudes(state, vector_1, vector_2):
    alpha_0 = inner(vector_1 , state)
    alpha_2 = inner(vector_2 , state)

    amps = np.array([alpha_0 , alpha_2])
    return amps

def major_decomp(state, particle_hole):
   # This function takes in a positive energy, fermion 'state' and returns the two Majorana wavefunctions it is constructed from.
   # We do this by superposing the state and it's hole conjugate, which is calculated by application of the operator ph.
   # If z = a - ib denotes a positive energy fermion state. Then z + z* is the real part and 1j*(z - z*) is the imaginary part.
   real = (state + ph_me(state, particle_hole ))/ np.sqrt(2)
   imaginary = 1j*(state - ph_me(state , particle_hole))/np.sqrt(2)

   return real , imaginary

def ph_me(vec , particle_hole):
    return particle_hole.dot(np.conjugate(vec))

# Berry phase calculator:
def Berry_new(phase_diff , no_points , length , width , lattice_constant, Magn):
    V = np.zeros(shape = (4*int(length/lattice_constant)*int(width/lattice_constant) , no_points) , dtype = np.complex128)
    norms = np.zeros(no_points)
    norms_V = np.zeros(no_points)
    phases = np.linspace(0 , phase_diff , no_points)
    evals = np.zeros(shape = (6,no_points) , dtype = np.complex128)
    for i in tqdm(range(no_points)):
        parameters = dict(a = lattice_constant , L = length , W = width , d = 1 , flux = 2 , offset = phases[i] , A =1 , B= 0.5 , C= Magn  , D = 0.75 , u = 0)
        system = make_josephson_junction(**parameters)
        eigenvalues, eigenvectors = generalised_solver(system)
        V[: , i] = eigenvectors[: , 0]
        evals[:, i]  = eigenvalues
        norms[i] = eigenvectors[: , 0].conj() @ eigenvectors[: , 0]
        norms_V[i] = V[: , i].conj() @ V[: , i]

    # Calculating loop of inner products using rolls:
    prods = np.roll(V.conj().T , shift = -1 , axis = 0) @ V
    # Loop of inner products is the product of all diagonal elements:\
    loop = prods.diagonal().prod()
    # Berry phase:  
    berry = - np.imag(np.log(loop))
    return berry, V, evals

def major_Berry(phase_diff , no_points , length , width , lattice_constant, Magn):
    # Constructing the particle-hole operator for later:
    dim = int(length/lattice_constant)*int(width/lattice_constant)
    # ph = np.kron(np.eye(dim),np.kron(sy , sy))
    Big_eye = sparse.csr_matrix(np.eye(dim))
    unitary = np.kron(sy , sy)
    ph = sparse.kron(Big_eye , unitary)

    # This is just like Berry_new except it splits the lowest energy state into two Majorana wavefunctions and calculates the Berry phases accrued by each separately.
    V_r = np.zeros(shape = (4*dim , no_points) , dtype = np.complex128)
    V_i = np.zeros(shape = (4*dim , no_points) , dtype = np.complex128)
    phases = np.linspace(0 , phase_diff , no_points)
    evals = np.zeros(shape = (10,no_points) , dtype = np.complex128)

    print('Calculating Majorana wavefunctions:')
    for i in tqdm(range(no_points)):
        parameters = dict(a = lattice_constant , L = length , W = width , d = 1 , flux = 2 , offset = phases[i] , A =1 , B= 0.5 , C= Magn  , D = 0.75 , u = 0)
        system = make_josephson_junction(**parameters)
        eigenvalues, eigenvectors = generalised_solver(system, 10)
        # For the purpose of constructing the Majorana wavefunctions it is enough if we just take the lowest *positive* energy wavefunction so:
        eigenvectors = eigenvectors[: , eigenvalues > 0]
        V_r[: , i], V_i[: , i] = major_decomp(eigenvectors[: , 0], ph)
        evals[:, i]  = eigenvalues
    print('Complete :)')

    # Berry phase of real bit:

    # Calculating loop of inner products using rolls:
    prods_r = np.roll(V_r.conj().T , shift = -1 , axis = 0) @ V_r
    # Loop of inner products is the product of all diagonal elements:\
    loop_r = prods_r.diagonal().prod()
    # Berry phase:  
    berry_r = - np.imag(np.log(loop_r))

    # Berry phase of imaginary component:

    # Calculating loop of inner products using rolls:
    prods_i = np.roll(V_i.conj().T , shift = -1 , axis = 0) @ V_i
    # Loop of inner products is the product of all diagonal elements:\
    loop_i = prods_i.diagonal().prod()
    # Berry phase:  
    berry_i = - np.imag(np.log(loop_i))

    return berry_r , berry_i, V_r, V_i, evals

def non_abelian_major_Berry(phase_diff , no_points , length , width , lattice_constant, Magn , Delta, finite_diff, cn):    
    # Description: This function calculates the non-Abelian Berry holonomy for Majorana wavefunctions:

    # Inputs:
    #phase_diff: Total phase difference across the junction.
    #no_points: No. points along the loop.
    #length: Length of the Junction.
    #width: width of the system.
    #lattice_constant: Lattice constant.
    #Magn: Magnetisation parameter.
    # Delta: Magnitude of the Gap function.
    #finite_diff: A string which determines the finite difference scheme.
    #cn: Bool which controls whether evolution is implemented using the crank nicholson scheme.  

    central_difference = False

    if finite_diff == 'fd':
        sign = 1
    if finite_diff == 'bd':
        sign = -1
    if finite_diff == 'cd':
        central_difference = True
        sign = 1

    # Constructing the particle-hole operator for later:
    dim = int(length/lattice_constant)*int(width/lattice_constant)
    # ph = np.kron(np.eye(dim),np.kron(sy , sy))
    Big_eye = sparse.csr_matrix(np.eye(dim))
    unitary = np.kron(sy , sy)
    ph = sparse.kron(Big_eye , unitary)
    # This is just like Berry_new except it splits the lowest energy state into two Majorana wavefunctions and calculates the Berry phases accrued by each separately.
    V_r = np.zeros(shape = (4*dim , no_points) , dtype = np.complex128)
    V_i = np.zeros(shape = (4*dim , no_points) , dtype = np.complex128)
    phases = np.linspace(0 , phase_diff , no_points)
    evals = np.zeros(shape = (10,no_points) , dtype = np.complex128)

    print('Calculating Majorana wavefunctions:')
    for i in tqdm(range(no_points)):
        parameters = dict(a = lattice_constant , L = length , W = width , d = 1 , flux = 2 , offset = phases[i] , A =1 , B= 0.5 , C= Magn  , D = Delta , u = 0)
        system = make_josephson_junction(**parameters)
        eigenvalues, eigenvectors = generalised_solver(system, 10)
        # For the purpose of constructing the Majorana wavefunctions it is enough if we just take the lowest *positive* energy wavefunction so:
        eigenvectors = eigenvectors[: , eigenvalues > 0]
        V_r[: , i], V_i[: , i] = major_decomp(eigenvectors[: , 0], ph)
        evals[:, i]  = eigenvalues
    print('Complete :)')

    print('Calculating Berry Holonomies:')

    def naive_connection(step):
        F = np.zeros(shape = (4*dim , 2) , dtype = np.complex128)
        F[: , 0] = V_r[: , step]
        F[: , 1] = V_i[: , step]

        G = np.zeros(shape = (4*dim , 2) , dtype = np.complex128)
        G[: , 0] = np.conjugate(V_r[: , (step+sign*1)%no_points])
        G[: , 1] = np.conjugate(V_i[: , (step+sign*1)%no_points])
        link_matrix = (G.T)@F

        if central_difference == True:
            H = np.zeros(shape = (4*dim , 2) , dtype = np.complex128)
            H[: , 0] = np.conjugate(V_r[: , (step-sign*1)%no_points])
            H[: , 1] = np.conjugate(V_i[: , (step-sign*1)%no_points])
            link_matrix -= (H.T)@F # Subtracting the backward difference too for central difference method.
            link_matrix *= 0.5 # For the central difference scheme, the step size is twice as long.
        return link_matrix 

    def crank_nicholson(ell):
        forward = np.eye(2) - 0.5*ell
        ell_dagger = np.conjugate(ell.T)
        backward = np.linalg.inv(np.eye(2) + 0.5*ell_dagger) 
        return backward @ forward

    def exp_expansion(ell):
        if (finite_diff == 'fd') or (finite_diff == 'bd'):
            return ell
        if finite_diff == 'cd':
            return np.eye(2) - ell

    # Lists to store intermediate data:
    links = [] 
    running_prod = []
    for i in tqdm(range(no_points)):
        lmatrix = naive_connection(i)
        ith_link = exp_expansion(lmatrix)
        if cn == True:
            ith_link = crank_nicholson(lmatrix)
        if i == 0:
            product = ith_link 
        if i > 0:
            product = ith_link @ product
        links.append(ith_link)
        running_prod.append(product)

    # Calculating individual Abelian phases:
    # Berry phase of real bit:
    
    # Calculating loop of inner products using rolls:
    prods_r = np.roll(V_r.conj().T , shift = -1 , axis = 0) @ V_r
    # Loop of inner products is the product of all diagonal elements:\
    loop_r = prods_r.diagonal().prod()
    # Berry phase:  
    berry_r = - np.imag(np.log(loop_r))

    # Berry phase of imaginary component:

    # Calculating loop of inner products using rolls:
    prods_i = np.roll(V_i.conj().T , shift = -1 , axis = 0) @ V_i
    # Loop of inner products is the product of all diagonal elements:\
    loop_i = prods_i.diagonal().prod()
    # Berry phase:  
    berry_i = - np.imag(np.log(loop_i))
    print('Complete :) ')
    return product, berry_r, berry_i, links, running_prod, V_r, V_i , evals
