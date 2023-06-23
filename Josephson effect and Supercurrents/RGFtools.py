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
    # Eigenvalues corresponding to states that decay in the positive x direction:
    pos_e_evals = eigenvalues[(np.abs(eigenvalues) - 1) < -tol]
    # Eigenvalues corresponding to states that evanesce in the negative x direction:
    neg_e_evals = eigenvalues[(np.abs(eigenvalues) - 1) > tol]
    # Eigenvalues corresponding to propagating states in the +ve x direction:
    propagatingstates = eigenvalues[np.abs((np.abs(eigenvalues) - 1)) <= tol]
    pos_p_evals = propagatingstates[np.angle(propagatingstates) > 0]
    # Eigenvalues corresponding to propagating states in the -ve x direction:
    neg_p_evals = propagatingstates[np.angle(propagatingstates) < 0]

    # Checking that the lengths of pos_e_evals , neg_e_evals , pos_p_evals , neg_p_evals sum up to len(evals)
    if (len(pos_e_evals) + len(pos_p_evals) + len(neg_e_evals) + len(neg_p_evals)) != len(eigenvalues):
        raise Exception("The number of evanescent and propagating states in the +ve and -ve x direction does not match the length of eigenvalues array. Change tolerance!")

    #Eigenvectors that evanesce in the positive x direction:
    pos_evanesce = eigenvectors[: , (np.abs(eigenvalues) - 1) < -tol]
    #Eigenvectors that evanesce in the negative x direction:
    neg_evanesce = eigenvectors[: , (np.abs(eigenvalues) - 1) > tol]
    #Eigenvectors that propagate in the positive x direction:
    propagatingeigenvectors = eigenvectors[: , np.abs((np.abs(eigenvalues) - 1)) <= tol]
    pos_prop = propagatingeigenvectors[: , np.angle(propagatingstates) > 0]
    neg_prop = propagatingeigenvectors[: , np.angle(propagatingstates) < 0]
    
    list_of_eigenvalues= [pos_p_evals , neg_p_evals , pos_e_evals , neg_e_evals]
    
    return pos_prop , neg_prop , pos_evanesce , neg_evanesce , list_of_eigenvalues

def calculate_transfer_matrices(slice , params ):
    energy = params.energy
    M00 = np.linalg.inv(T(slice+1 , -1, params))@(energy*np.identity(2*params.W) -  h_0(slice , params)) # <- Calculating the Hamiltonian at slice.
    M01 = -np.linalg.inv(T(slice+1 , -1, params))@T(slice , +1 , params)
    M10 = np.identity(2*params.W)
    M11 = np.zeros(shape = (2*params.W , 2*params.W))

    M = np.block([[M00 , M01],[M10 , M11]]) #<- Matrix to diagonalise for propagating modes in lead
    evals , evecs = np.linalg.eig(M)
    
    # Matrix M is not symmetric, so vectors are not necessarily orthonormal.
    # Let us orthogonalise them:
    evecs_orth = orth(evecs)

    pos_prop , neg_prop , pos_evanesce , neg_evanesce , list_of_eigenvalues = sorting_modes(evals , evecs_orth  , tol = 1e-4)
    # First I am going to glue together all modes that evanesce/propagate in the positive x-direction:
    pos_modes = np.hstack((pos_prop , pos_evanesce))
    neg_modes = np.hstack((neg_prop , neg_evanesce))

    # The U(\pm)-matrices consisten of amplitudes on the j = 0 slice. So we take only the first half rows:
    U_pos = pos_modes[0:int(pos_modes.shape[0]/2) , :]
    U_neg = neg_modes[0:int(neg_modes.shape[0]/2) , :]

    # The \Lambda(\pm) matrix comprises of all the corresponding eigenvalues:
    Lambda_pos = np.diag(np.hstack((list_of_eigenvalues[0] ,list_of_eigenvalues[2])))
    Lambda_neg = np.diag(np.hstack((list_of_eigenvalues[1] , list_of_eigenvalues[3])))

    # Construct the F(\pm) transfer matrices:
    F_pos = U_pos @ Lambda_pos @ np.linalg.inv(U_pos)
    F_neg = U_neg @ Lambda_neg @ np.linalg.inv(U_neg)

    # This is for diagnostic purposes:
    debugdict = {'U_pos' : U_pos , 'U_neg' : U_neg , 'Lambda_pos' : Lambda_pos , 'Lambda_neg': Lambda_neg
                 , 'pos_prop' : pos_prop , 'neg_prop' : neg_prop , 'pos_evanesce' : pos_evanesce , 'neg_evanesce' : neg_evanesce
                 ,'list_of_eigenvalues' : list_of_eigenvalues , 'evals': evals , 'evecs': evecs}
    
    return F_pos , F_neg , debugdict 