import numpy as np
import tinyarray as tiny
from scipy.linalg import orth
sx = tiny.array([[0 , 1] , [1 , 0]])
sy = tiny.array([[0 , -1j] , [1j , 0]])
sz = tiny.array([[1 , 0] , [0 , -1]])
I = tiny.array([[1 , 0] , [0 , 1]])
t_plus = 0.5*(np.kron(sx , I) + 1j*np.kron(sy , I))
t_minus = t_plus.T
phs = np.kron(sy , sy)
import scipy

def h_0(j , params):
    # This function constructs the Hamiltonian for sites along the jth slice. It does not include hoppings between neighbouring slices.
    # We will use the gauge A = By \hat{x}
    if j < 1:
        onsitematrix = 4*params.t*sz - params.mu*sz + params.Delta*sx

    if j >= 1 and j <=params.L:
        onsitematrix = 4*params.t*sz -params.mu*sz
    
    if j > params.L:
        onsitematrix = 4*params.t*sz - params.mu*sz + params.Delta*np.exp(1j*params.phase)*np.array([[0 , 1] ,[0 , 0]]) + params.Delta*np.exp(-1j*params.phase)*np.array([[0 , 0] ,[1 , 0]])
    
    off_diagonal_elements = np.diag( np.ones(params.W-1) , k = 1) + np.diag( np.ones(params.W-1) , k = -1)
    h_hops = np.kron(off_diagonal_elements , -params.t*sz)

    h_onsites = np.kron(np.identity(params.W) , onsitematrix)
    h_0 = h_onsites + h_hops
    return h_0

def T(j, pm , params):
    peierlphase = 2*np.pi*params.flux / ((params.L - 1)*params.W)
    if j <= 1 or j >= params.L + 1:
        return np.kron(np.identity(params.W) , -params.t*sz)
    if j >= 2 and j <= params.L:
        peierls_e = pm*(peierlphase)*np.arange(params.W) # array of peierl phases for electrons
        peierls_h = -peierls_e # array of peierl phases for holes.
        peierls_interweaved = np.exp(1j*np.vstack((peierls_e , peierls_h)).reshape((-1 , ) , order = 'F')) #<- talking elementwise exponential.
        peierls_bdg = np.diag(peierls_interweaved) # Peierls phases as the appear in the BdG matrix
        return np.kron(np.identity(params.W) , -params.t*sz)@peierls_bdg

def sorting_modes(eigenvalues , eigenvectors , tol = 1e-4):
    '''
    This function sorts modes into those that evanesce in the positive-x and negative-x directions
    and those that propagate in the +ve and -ve x direction.
    Inputs:
    eigenvalues: array_like, vector of eigenvalues.
    eigenvectors: ndarray , matrix of eigenvectors where columns 'i' to eigenvalue 'i'.
    tol: float , optional.  Tolerance to determine if eigenvalue is unit magnitude. If not set manually, default precision is 1e-4.
        WARNING: If tolerance is smaller than numerical precision, the function might "miss" all the propagating modes.
    Returns:
    pos_evanesce , neg_evanesce: ndarray, Eigenvectors that evanesce in the positive and negative x-direction respectively.
    pos_prop , neg_prop: ndarray, Eigenvectors that evanesce in the positive and negative x-direction respectively.
    list_of_eigenvalues:list, List of arrays [pos_p_evals , neg_p_evals , pos_e_evals , neg_e_evals] with eigenvalues sorted according to their propagation type.
    '''
    # each eigenvector is formed of amplitudes (c_{i} , c_{i-1}). We want the lower block:
    selected_indices = int(eigenvectors.shape[0]/2)
    vectors = eigenvectors[selected_indices : , :]

    # We now want a matrix of norms. The column vectors in vectors may not necessarily be orthogonal but this is fine.
    norms = np.diagonal(np.conj(vectors.T) @ vectors)
    normalisation_factors = 1 / np.sqrt(norms)
    normalisation_factor_matrix = np.diag(normalisation_factors)

    normalised_vectors = vectors @ normalisation_factor_matrix

    # Eigenvalues corresponding to states that decay in the positive x direction:
    pos_e_evals = eigenvalues[(np.abs(eigenvalues) - 1) < -params.tol]
    # Eigenvalues corresponding to states that evanesce in the negative x direction:
    neg_e_evals = eigenvalues[(np.abs(eigenvalues) - 1) > params.tol]

    # Compute the group velocity for each state:
    group_velocities = np.zeros(vectors.shape[1])
    for i in range(vectors.shape[1]):
        group_velocities[i] = group_vel(normalised_vectors[: , i], eigenvalues[i] , slice , params)

    # Sorting propagating modes according to sign of group velocity:
    p_conditions = (np.abs((np.abs(eigenvalues) - 1)) <= params.tol) & (group_velocities > 0)
    n_conditions =(np.abs((np.abs(eigenvalues) - 1)) <= params.tol) & (group_velocities < 0 )

    pos_p_evals = eigenvalues[p_conditions]
    neg_p_evals = eigenvalues[n_conditions]

    # Sorting normalised eigenvectors in the same way:
    pos_prop = normalised_vectors[: , p_conditions]
    neg_prop = normalised_vectors[: , n_conditions]

    pos_evanesce = normalised_vectors[: , (np.abs(eigenvalues) - 1) < -params.tol]
    neg_evanesce = normalised_vectors[: , (np.abs(eigenvalues) - 1) > params.tol]
    
    list_of_eigenvalues = [pos_p_evals , neg_p_evals , pos_e_evals , neg_e_evals]

    # group_velocities for propgating states only:
    g_vel_p = group_velocities[p_conditions]
    g_vel_n = group_velocities[n_conditions]
     
    return pos_prop , neg_prop , pos_evanesce , neg_evanesce , list_of_eigenvalues , [group_velocities , g_vel_p , g_vel_n]

def generalised_eigenvalue_problem(slice , params):
    energy = params.energy
    M00 = np.linalg.inv(T(slice+1 , -1, params))@(energy*np.identity(params.W) -  h_0(slice , params)) # <- Calculating the Hamiltonian at slice.
    M01 = -np.linalg.inv(T(slice+1 , -1, params))@T(slice , +1 , params)
    M10 = np.identity(params.W)
    M11 = np.zeros(shape = (params.W , params.W))

    M = np.block([[M00 , M01],[M10 , M11]]) #<- Matrix to diagonalise for propagating modes in lead

    # evals are bloch factors, evecs are transverse mode eigenvfunctions for sites: (slice , slice -1)^{T}
    evals , evecs = scipy.linalg.eig(M)

    pos_prop , neg_prop , pos_evanesce , neg_evanesce , list_of_eigenvalues , gv = sorting_modes(evals , evecs ,params)

    # Constructing U(±), matrices of right and left going eigenvectors:
    U_pos = np.hstack((pos_prop , pos_evanesce))
    U_neg = np.hstack((neg_prop , neg_evanesce)) 

    # Constructing A(±) , diagonal matrix of bloch factors:
    Lambda_pos = np.diag(np.hstack((list_of_eigenvalues[0] ,list_of_eigenvalues[2])))
    Lambda_neg = np.diag(np.hstack((list_of_eigenvalues[1] , list_of_eigenvalues[3])))

    # Construct the F(\pm) transfer matrices:
    F_pos = U_pos @ Lambda_pos @ np.linalg.inv(U_pos)
    F_neg = U_neg @ Lambda_neg @ np.linalg.inv(U_neg)

    F_pos_prop_only = pos_prop @ np.diag(list_of_eigenvalues[0])@np.linalg.pinv(pos_prop)
    F_neg_prop_only = neg_prop @ np.diag(list_of_eigenvalues[1])@np.linalg.pinv(neg_prop)

        # This is for diagnostic purposes:
    debugdict = {'U_pos' : U_pos , 'U_neg' : U_neg , 'Lambda_pos' : Lambda_pos , 'Lambda_neg': Lambda_neg
                 , 'pos_prop' : pos_prop , 'neg_prop' : neg_prop , 'pos_evanesce' : pos_evanesce , 'neg_evanesce' : neg_evanesce
                 ,'list_of_eigenvalues' : list_of_eigenvalues , 'group_velocities' : gv, 'evals': evals , 'evecs': evecs , 
                 'F_pos_prop': F_pos_prop_only , 'F_neg_prop' : F_neg_prop_only}

    return F_pos , F_neg , debugdict 

def group_vel(vector , bloch_factor, slice, params):
    """
    Given an eigenvector (\vec{u}_{n}) and bloch_factor (\lambda_{n}), this function computes the group velocity of the state.
    We use Equation 5 in https://arxiv.org/pdf/cond-mat/0501609.pdf

    Arguments:
    - vector: nd-array to compute group velocity of.
    - bloch_factor: complex128, bloch factor associated with that state.
    - slice: int , slice index.
    - params: system parameters. 
    Returns:
     Group velocity of the wavefunction.
    """
    product = bloch_factor*(np.conjugate(vector.T))@T(slice , -1 , params)@vector
    prefactor = -(2*params.a)*2*np.pi

    return np.imag(prefactor*product)

def RGF_left_sweep(lead_indices , params):
    """
    Computes the left Greens function for the system.
    Uses the recursive Greens function algorithm to compute diagonal and off-diagonal Greens functions between the two leads.

    Arguments:
    lead_indices: list , indices of lead slices that are connected to the scattering region.
    params: system parameters 

    returns:
    [GF_diagonal , GF_off_diagonal] : list , diagonal (n , n) and off-diagonal (n , 0) left Greens functions for the system, for n: {0 , 1 , ... , L+1},
                                       where L is the system size in the x-direction.
    """
    # Indices of site in left and right lead that are connected to the scattering region:
    #  Lead  *  Scattering  * Lead
    # -L-L-L-L-S-S-S...-S-S-R-R-R-R-
    #       -0-1-2-3.....-L-L+1
    
    slice_left = lead_indices[0]
    slice_right = lead_indices[1]

    F_pos_left , F_neg_left , debug_left = generalised_eigenvalue_problem(slice_left - 1 , params)
    F_pos_right , F_neg_right , debug_right = generalised_eigenvalue_problem(slice_right + 1 , params)
    
    H_start_tilde = h_0(slice_left , params) + T(slice_left , +1 , params)@scipy.linalg.inv(F_neg_left)
    H_end_tilde = h_0(slice_right , params) + T(slice_right+1 , -1 , params)@F_pos_right

    # Number of slices in the scattering region + adjacent slices in lead:
    no_steps = params.L + 2
    
     # Storing tilde Hamiltonians in an array:
    H_tildes = np.zeros(shape = ( H_start_tilde.shape[0] , H_start_tilde.shape[0] , no_steps))
    H_tildes[: , : , 0] = H_start_tilde
    H_tildes[: , : ,-1] = H_end_tilde
    for j in range(1,params.L + 1):
        H_tildes[: , : , j] = h_0(j , params)

    # Source term at the left lead:
    source_term = T(slice_left , +1 , params) @ (scipy.linalg.inv(F_pos_left) - scipy.linalg.inv(F_neg_left))

    # Recursive step:
    G_start = scipy.linalg.inv(params.energy*np.identity(params.W) - H_start_tilde)    
    diag_greensfunctions = np.zeros(shape = (params.W , params.W , no_steps) , dtype = np.complex128)
    off_diag_greensfunctions = np.zeros(shape = (params.W , params.W , no_steps), dtype = np.complex128)
    
    # Initial condition:
    diag_greensfunctions[: , : , 0] = G_start
    off_diag_greensfunctions[: , : , 0] = G_start

    for i in range(no_steps):
        if i > 0:
            matrix = (params.energy*np.identity(params.W)  - H_tildes[: , : , i] -
                            (T(i , +1 , params) @ diag_greensfunctions[: , : , i-1] @ T(i , -1 , params)) )
            diag_greensfunctions[: , : , i] = scipy.linalg.inv(matrix)
            off_diag_greensfunctions[: , : , i] = diag_greensfunctions[: , : , i] @ T(i , +1 , params) @ off_diag_greensfunctions[: , : , i-1]
    
    return diag_greensfunctions , off_diag_greensfunctions

def transmission_matrix_left(lead_indices , diag_greensfunctions , off_diag_greensfunctions , params ):
    """
    Computes the normalized transmission matrix / reflectionmatrix from the left lead to the right lead. Included in the library for diagnostic purposes.
    Arguments: 
    lead_indices: list or ndarray, j-indices corresponding to each lead.
    diag_greensfunctions: ndarray, 3D array of diagonal left Greens functions G^{L}_{n,n}. The last index corresponds to the slice 'n'.
    off_diag_greensfunctiosn: ndarray, 3D array of diagonal left Greens functions G^{L}_{n , 0}. The last index corresponds to the slice 'n'.

    Returns:
    transmission: normalized transmission matrix.
    unphysical_transmission: unphysical transmission matrix that includes evanescent modes (included for diagnostics)
    """
    slice_left = lead_indices[0]
    slice_right = lead_indices[1]
    F_pos_left , F_neg_left , debug_left = generalised_eigenvalue_problem(slice_left - 1 , params)
    F_pos_right , F_neg_right , debug_right = generalised_eigenvalue_problem(slice_right + 1 , params)

    # Source term at the left lead:
    source_term = T(slice_left , +1 , params) @ (scipy.linalg.inv(F_pos_left) - scipy.linalg.inv(F_neg_left))

    U_pos_left = debug_left['U_pos']
    U_pos_right = debug_right['U_neg']
    
    # Non-physical transmission matrix (includes matrix elements between evanescent waves too, so not physical):  
    unphysical_transmission = scipy.linalg.inv(U_pos_right)@off_diag_greensfunctions[: , : , -1]@source_term@U_pos_left

    # For the physical transmission matrix, we must compute matrix-elements only between propagating states:
    U_pos_prop_left = debug_left['pos_prop']
    U_pos_prop_right = debug_right['pos_prop']

    transmission = scipy.linalg.pinv(U_pos_prop_right) @  off_diag_greensfunctions[: , : , -1] @ source_term  @ U_pos_prop_left

    # Normalise with respect to group_velocities of propagating modes in each lead:
    g_vel_left = debug_left['group_velocities'][1]
    g_vel_right = debug_right['group_velocities'][1]

    normalisation_matrix = np.sqrt(np.outer(g_vel_right , 1/g_vel_left))

    transmission = normalisation_matrix * transmission

    return transmission , unphysical_transmission