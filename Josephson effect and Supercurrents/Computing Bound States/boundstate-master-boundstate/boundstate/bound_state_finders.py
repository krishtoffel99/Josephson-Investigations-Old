"""
*** Copied from _standard due to annoying import error***
Compute bound state using SVD.

Use 'scan_singular_values' to have a guest of energies of bound states.
Bound stats are located where singular value = 0.

Use 'finder' to find the exact energy of bound states in a small range of
energies.

Use 'solver' to obtain the wave-function of the bound states at a precise
energy.
"""

from __future__ import division
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize_scalar
np.set_printoptions(precision=16, suppress=1)
import scipy.sparse as sp


def finder(syst, e_range, args=(), sparse=False,
           tol=1e-6, sigma=None, sing_values=4, uv=False):
    """
    Finds the minimum of the singular values of L in e_range.
    If the singular value is smaller than tol, then it returns
    the energy of the bound state.
    Parameters:
    e_range: tuple countaining the minimum and the maximum of the
             energy range

    returns:
    E: energy of the bound state if minimal singular value is smaller
        than tol
    None otherwise
    tol: float, if the minimal singular value found by minimize_scalar is
        smaller than tol, then it consider you found a bound state
    tol: float, function minimize_scalar stops only if the minimum of the
        previous step is closer than tol (or something like that). Should
        be smaller than tol
    """
    H_s, H_leads, V_leads, transf = extract_kwant_matrices(syst, args=args,
                                                           sparse=sparse)
    return _finder(H_s, H_leads, V_leads, transf, e_range, sparse,
                   tol, sigma, sing_values, uv)


def _finder(H_s, H_leads, V_leads, transf, e_range, sparse=False,
            tol=1e-6, sigma=None, sing_values=4, uv=False):
    """
    Finds the minimum of the singular values of L in e_range.
    If the singular value is smaller than tol, then it returns
    the energy of the bound state.
    Parameters:
    e_range: tuple countaining the minimum and the maximum of the
             energy range

    returns:
    E: energy of the bound state if minimal singular value is smaller
        than tol
    None otherwise
    tol: float, if the minimal singular value found by minimize_scalar is
        smaller than tol, then it consider you found a bound state
    tol: float, function minimize_scalar stops only if the minimum of the
        previous step is closer than tol (or something like that). Should
        be smaller than tol
    """

    def dumb(e):
        s = min(_singular_values(H_s, H_leads, V_leads, transf, e,
                                 sparse=sparse, uv=False, sigma=sigma,
                                 sing_values=sing_values))
        return s

    minimum = minimize_scalar(dumb, method='Bounded', bounds=e_range,
                              options={'xatol': tol})
    if minimum.fun < tol:
        # if the last singular value is smaller than tol,
        # then there is a bound state.
        E = minimum.x
        if not uv:
            return E
        else:
            S, Wh, L_out, X_out = _singular_values(H_s, H_leads, V_leads,
                                                   transf, E, sparse=sparse,
                                                   uv=True, sigma=sigma,
                                                   sing_values=sing_values)
            return E, S, Wh, L_out, X_out
    else:
        return None


def solver(syst, E, args=(), tol=1e-4, sparse=False, sigma=None):
    """
    Returns the elements to compute the wavefunction of a bound state
    Raise an error if there is no bound state (singular value > tol)
    Parameters:
    transf: numpy array or sparse csr matrix. V.dot(transf) gives the
    hopping matrix between the scattering region and the leads.
    """
    H_s, H_leads, V_leads, transf = extract_kwant_matrices(syst, args=args,
                                                           sparse=sparse)
    return _solver(H_s, H_leads, V_leads, transf, E, tol, sparse, sigma)


def _solver(H_s, H_leads, V_leads, transf, E, tol=1e-4, sparse=False,
            sigma=None):
    """
    Returns the elements to compute the wavefunction of a bound state
    Raise an error if there is no bound state (singular value > tol)
    Parameters:
    transf: numpy array or sparse csr matrix. V.dot(transf) gives the
    hopping matrix between the scattering region and the leads.
    """
    S, Wh, L_leads, X_out_leads = _singular_values(H_s, H_leads, V_leads,
                                                   transf, E, sparse=sparse,
                                                   uv=True, sigma=sigma)
    assert(min(S) < tol), 'The energy does not match\
          the bound state energy, {0}'.format(min(S))
    N = H_s.shape[0]
    W = Wh.conj().T
    return compute_wf(S, W, L_leads, X_out_leads, N, eps=tol)


def hopping_svd(V_leads, tol=1e-4):
    """
    Check if one or several hopping matrices in the lead are not
    invertible and replace them by their SVD in the former case.
    """
    DBh_leads = []
    for V in V_leads:
        _, D, Bh = la.svd(V)
        # keep only the vectors assiocated with non zero singular value
        non_zeros = sum(D > tol)
        D = np.diag(D[:non_zeros])
        Bh = Bh[:non_zeros, :]
        DBh = np.dot(D, Bh)
        DBh_leads.append(DBh)

    return DBh_leads


def singular_values(syst, E, args=(), sparse=False,
                    uv=True, sing_values=4, tol=1e-4, sigma=None,
                    need_orth=True):
    """
    Solver that returns the singular values of the SVD of
    the left hand side of the bound state equation (L in the notes)

    Parameters:
    H_s: Hamiltonian of the central system
    H_leads: tuple,
             countains the onsite matrices of the unit cells of the leads
    V_leads: tuple, countains the hopping matrices between the unit
        cells of every lead
    uv: Whether to compute the the full SVD of the matrix L
        or only the singular values
    returns:
    if uv is false: the singular values of the matrix L
    else:
    S: The singular values of L
    Wh: The right matrix of the svd decomposition
    L: array that countains the evanescent outgoing lambdas
    X_out: Columns are the corresponding evanescent wavefunctions
    """
    H_s, H_leads, V_leads, transf = extract_kwant_matrices(syst, args=args,
                                                           sparse=sparse)
    return _singular_values(H_s, H_leads, V_leads, transf, E, sparse,
                            uv, sing_values, tol, sigma,
                            need_orth)


def _singular_values(H_s, H_leads, V_leads, transf, E, sparse=False,
                     uv=True, sing_values=4, tol=1e-4, sigma=None,
                     need_orth=True):
    """
    Solver that returns the singular values of the SVD of
    the left hand side of the bound state equation (L in the notes)

    Parameters:
    H_s: Hamiltonian of the central system
    H_leads: tuple,
             countains the onsite matrices of the unit cells of the leads
    V_leads: tuple, countains the hopping matrices between the unit
        cells of every lead
    uv: Whether to compute the the full SVD of the matrix L
        or only the singular values
    returns:
    if uv is false: the singular values of the matrix L
    else:
    S: The singular values of L
    Wh: The right matrix of the svd decomposition
    L: array that countains the evanescent outgoing lambdas
    X_out: Columns are the corresponding evanescent wavefunctions
    """
    small_num = np.finfo(float).eps * 1e2
    L_out_leads, X_out_leads = leads_modes(H_leads, V_leads, E, tol=tol)

    lhs = setup_lhs(H_s, V_leads, transf, L_out_leads, X_out_leads, E,
                    sparse=sparse)
    if sparse:
        B = sp.coo_matrix.dot(lhs.conj().T, lhs)
        if sigma is None:
            # First compute the biggest eigenvalue
            max_eval = sp.linalg.eigsh(B, k=1, which='LM',
                                       return_eigenvectors=False)[0]
            # shift the matrix so that the lowest eigenvalue
            # become the biggest in magnitude
            B = -B + 2 * max_eval * sp.eye(B.shape[0])
        if not uv:
            try:
                evals = sp.linalg.eigsh(B, return_eigenvectors=False,
                                        which='LM', k=sing_values, sigma=sigma)
            except RuntimeError:  # if the singular value is exactly zero
                evals = sp.linalg.eigsh(B + tol*sp.eye(B.shape[0]),
                                        which='LM', return_eigenvectors=False,
                                        k=sing_values, sigma=sigma)
                evals -= tol
            if sigma is None:  # that is, use the shift method
                return np.sqrt(2 * max_eval - evals)
            else:  # use of the shift invert method
                return np.sqrt(evals)
        else:
            try:
                evals, evecs = sp.linalg.eigsh(B, return_eigenvectors=True,
                                               k=sing_values, sigma=sigma,
                                               which='LM')
            except RuntimeError:
                evals, evecs = sp.linalg.eigsh(B + tol*sp.eye(B.shape[0]),
                                               which='LM',
                                               return_eigenvectors=True,
                                               k=sing_values, sigma=sigma)
                evals -= tol
            if sigma is None:  # shift method
                return (np.sqrt(2 * max_eval - evals), evecs.T, L_out_leads,
                        X_out_leads)
            else:  # shift invert method
                # If an eigenvalue is negative due to numerical
                # precision, then it is replaced by 0
                evals = np.clip(evals, 0, np.inf)
                return np.sqrt(evals), evecs.T, L_out_leads, X_out_leads
    else:
        if not uv:
            return la.svd(lhs, compute_uv=False)
        else:
            U, S, Wh = la.svd(lhs, compute_uv=True)
            return S, Wh, L_out_leads, X_out_leads


def setup_lhs(H_s, V_leads, transf, L_out_leads, X_out_leads, E, sparse=False):
    """
    Setup the left hand side of the equation to find bound states.
    """
    DBh_leads = hopping_svd(V_leads)
    DBh = block_diag(*DBh_leads)
    L_out = block_diag(*L_out_leads)
    X_out = block_diag(*X_out_leads)
    V = block_diag(*V_leads)

    if sparse:  # Uses either scipy.sparse if sparse of numpy if dense
        d = sp.csr_matrix.dot
        lin = sp
        DBh, V = sp.csr_matrix(DBh), sp.csr_matrix(V)
        L_out, X_out = sp.csr_matrix(L_out), sp.csr_matrix(X_out)
        # V is a csc matrix because it is transposed later
        # and it becomes a csr matrix
    else:
        d = np.dot
        lin = np

    Id_N = lin.eye(H_s.shape[0])
    DBh_ls = d(DBh, transf)
    V_ls = V.dot(transf)

    lhs = lin.hstack([H_s - E * Id_N, V_ls.conj().T.dot(d(X_out, L_out))])
    lhs = lin.vstack([lhs, lin.hstack([DBh_ls, -d(DBh, X_out)])])
    return lhs


def scan_singular_values(syst, e_min, e_max, args=(),
                         N=500, sparse=False, sigma=None, k=5):
    """Compute the singular values on a energy range.
    See 'singular_values' for details.
    """
    H_s, H_leads, V_leads, transf = extract_kwant_matrices(syst, args=args,
                                                           sparse=sparse)
    return _scan_singular_values(H_s, H_leads, V_leads, transf, e_min, e_max,
                                 args, N, sparse, sigma, k)


def _scan_singular_values(H_s, H_leads, V_leads, transf, e_min, e_max, args=(),
                          N=500, sparse=False, sigma=None, k=5):
    """Compute the singular values on a energy range.
    See 'singular_values' for details.
    """
    energies = np.linspace(e_min, e_max, N)
    if sparse:
        dumb = np.zeros(shape=(N, k))
    else:
        # at maximum the number of singular value is the dimension of H_s
        # + the dimension of all the leads together
        dumb = np.zeros(shape=(N, H_s.shape[0] +
                               sum(V.shape[0] for V in V_leads)))
        # the zeros are later replaced with NaN so they are not plotted.

    for i, e in enumerate(energies):
        s = _singular_values(H_s, H_leads, V_leads, transf, e, sparse,
                             uv=False, sigma=sigma, sing_values=k)

        dumb[i, :len(s)] = s
        dumb = fill_zero_with_nan(dumb)

    return (energies, dumb)

from kwant.physics.leads import unified_eigenproblem, setup_linsys
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) > 1)


def null_space(V, eps=1e-8):
    """
    Computes the projector of the range of V,
    Q such that VQ = 0, Q^2 = Q
    Parameters:
    V: 2D numpy array
    Returns
    Q: 2D numpy array, same dimensions as V
    """
    U, S, WH = la.svd(V)
    assert(np.allclose(np.conj(np.transpose(WH)).dot(WH),
           np.eye(WH.shape[0]))), 'SVD matrices not unitary'
    zero_eig_val = (S < eps)
    dim = len(S)
    Q = np.zeros(shape=V.shape, dtype=complex)
    for alpha in range(dim):
        if zero_eig_val[alpha]:
            x_alpha = WH[alpha, :]
            Q += np.dot(np.conj(x_alpha[:, None]), x_alpha[None, :])
    # two checks, optionnal
    # ~ assert(np.allclose(np.dot(Q, Q), Q)), 'Projector is not
    # idempotent (P**2 =/= P)'
    # ~ assert(np.allclose(np.dot(V, Q), np.zeros(shape=V.shape), atol=1e-4)),
    # 'Projection on the kernel is wrong'

    return Q


def fill_zero_with_nan(array):
    """
    Replace all elements equal to 0 by nan so that it is
    not plotted by pyplot
    """
    S = array.shape
    array = np.reshape(array, S[0] * S[1])
    for i, element in enumerate(array):
        if element != 0:
            array[i] = element
        else:
            array[i] = float('nan')
    return np.reshape(array, (S[0], S[1]))


def leads_modes(H_leads, V_leads, E, tol=1e-5, need_orth=True):
    """
    Compute the evanescent modes for every lead separetly
    """
    X_out_leads = []
    L_out_leads = []
    assert len(H_leads) == len(V_leads), 'Check the number of leads'

    for H, V in zip(H_leads, V_leads):  # compute the modes of every lead
        L_out, X_out = extract_out_modes(H, V, E, tol=tol, need_orth=need_orth)
        X_out_leads.append(X_out)
        L_out_leads.append(L_out)
    return L_out_leads, X_out_leads


def block_diag(*arrs):
    """
    Copied from old version of scipy, v0.8.0
    Create a block diagonal matrix from the provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Parameters
    ----------
    A, B, C, ... : array-like, up to 2D
        Input arrays.  A 1D array or array-like sequence with length n is
        treated as a 2D array with shape (1,n).

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
        same dtype as `A`.

    References
    ----------
    .. [1] Wikipedia, "Block matrix",
           http://en.wikipedia.org/wiki/Block_diagonal_matrix

    Examples
    --------
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> print(block_diag(A, B, C))
    [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 3 4 5 0]
     [0 0 6 7 8 0]
     [0 0 0 0 0 7]]
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                         "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


def extract_out_modes(H, V, E, tol=1e-6, return_mat=False, need_orth=True):
    """
    parameters:
    H: hamiltonian inside the unit cell of the lead
    V: hoppings between the cells
    E: Energy at which the mode is calculated
    need_orth: bool, if True it returns orthogonalized
                eigenvectors Phi_e, only necessary for degenerate evs

    returns:
    modes: a list of evanescent right going modes
    Normalization of X_out is such that max(X_out[:, i]) = 1
    """
    n = H.shape[0]
    Id = np.eye(n)
    H = H - E * Id
    V_dag = V.conj().T

    matrices, v, extract = setup_linsys(H, V, stabilization=(True, True))
    all_lmb, evanselect,\
        propselect, vec_gen, ord_schur = unified_eigenproblem(*(matrices))
    lmb_inv = all_lmb[evanselect]
    all_vecs = vec_gen(evanselect)
    Phi = np.zeros(shape=(n, len(lmb_inv)), dtype=complex)

    for i, l in enumerate(lmb_inv):
        phi = extract(all_vecs[:, i], l)
        phi /= np.sqrt(np.dot(phi.conj().T, phi))
        Phi[:, i] = phi

    Phi, arg_max, max_x = max_norm(Phi)

    # function that test if there is a degeneracy
    L = np.diag(1 / lmb_inv)
    if need_orth:
        if len(lmb_inv):
            all_dupl = find_groups(lmb_inv, tol=tol)
            for dupl in all_dupl:  # test is there are similar eigenvalues
                Phi[:, dupl], R = np.linalg.qr(Phi[:, dupl])
                temp = Phi[:, dupl].conj().T @ Phi[:, dupl]
        else:
            L = np.zeros(shape=(len(lmb_inv), 0), dtype=complex)

    if return_mat:
        return L, Phi, R
    else:
        return L, Phi


def find_groups(arr, tol=1e-6):
    """
    Return:
    all_dupl: list countaining lists with indices of equal eigenvalues
              up to the tolerance tol
    """
    # function that search if other_idx is already in all_dupl, modifies
    # all_dupl
    def already_dupl(all_dupl, idx, idx_min):
        for i, other_dupl in enumerate(all_dupl):
            if any([idx in other_dupl]):
                # a lready belong to another doublon, triplet...
                all_dupl[i].append(idx_min)
                return True
        return False

    def is_there_dupl(item, arr, idx):
        idx_min = np.argmin(abs(arr - item))
        if abs(arr[idx_min] - item) < tol:
            return idx + idx_min + 1
        return None

    all_dupl = []
    for idx, item in enumerate(arr[:-1]):
        dupl = []
        idx_min = is_there_dupl(item, arr[idx+1:], idx)
        if idx_min is None:
            pass
        else:
            # not already a duplicate
            if not already_dupl(all_dupl, idx, idx_min):
                dupl.extend((idx, idx_min))
                all_dupl.append(dupl)
    return all_dupl


def extract_kwant_matrices(sys, args=(), sparse=False):
    """
    Function that takes a finalized kwant system (with at least one lead)
    and returns the matrices defining the onsite and hoppings in the
    system.
    Parameters:
    sys: a kwant system with at least one lead
    sparse: whether to return sparse or dense matrices
    Returns:
    H_s: Hamiltonian of the scattering region
    H_leads: list countaining the hamiltonians of the unit cells of the
             leads
    V_leads: list countaining the hoppings between the unit cells of the
             leads
    transf: Rectangular matrix full of 0s and 1s that connects the lead
            and the scattering region. transf[i, j] = 1 only if the site
            i from the lead and the site j from the scattering region
            have a non-zero hopping.
    """
    assert(len(sys.leads) != 0), print('The system should countain at \
                                        least one lead')

    H_s = sys.hamiltonian_submatrix(args=args, sparse=sparse)
    H_leads = []
    V_leads = []
    coords = []
    for i, lead in enumerate(sys.leads):
        # H and V are not written in sparse at this point because the
        # mode solver needs a numpy array
        H_leads.append(sys.leads[i].cell_hamiltonian())
        V_leads.append(sys.leads[i].inter_cell_hopping())
        coords = np.concatenate((coords, sys.lead_interfaces[i]))

    norb = sys.hamiltonian_submatrix(args=args, sparse=sparse,
                                     return_norb=True)[1]
    offsets = np.empty(norb.shape[0] + 1, int)
    offsets[0] = 0
    offsets[1:] = np.cumsum(norb)
    ones = []
    for coord in coords:
        coord = int(coord)
        for i in range(offsets[coord], offsets[coord] + np.array(norb)[coord]):
            ones.append(i)

    if sparse:
        transf = sp.csr_matrix((np.ones(len(ones)), (range(len(ones)), ones)),
                               shape=(len(ones), sum(norb)))
    else:
        # most inefficient way to create a dense array, to change at some point
        transf = sp.csr_matrix((np.ones(len(ones)), (range(len(ones)), ones)),
                               shape=(len(ones), sum(norb))).toarray()

    return H_s, H_leads, V_leads, transf


def compute_wf(vals, vecs, L_leads, X_out_leads, scat_dim,
               eps=1e-4, schur_modes=False):
    """
    Extract and normalize the wavefunction from the eigenvectors of
    the lhs (H_eff) matrix.

    vals: eigenvalues of H_eff computed at the energy of the bound state
    vecs: eigenvectors of H_eff computed at the energy of the bound state
    L_leads: list of matrices with the evanescent modes in
    X_leads:
    scat_dim: int, number of sites in the scattering region
    schur_modes: if True, bound state is returned in schur basis, it is
                returned in the basis with Lambda diagonal otherwise
    """
    zero_schur = abs(vals) < eps
    # Psi_alpha_0 is the wavefunction of the bound states in the
    # system.
    psi_alpha_0, q_e = vecs[:scat_dim, zero_schur], vecs[scat_dim:, zero_schur]

    Q = block_diag(*X_out_leads)
    L_out = block_diag(*L_leads)

    # When modes are degenerated, Lambda is not diagonal anymore
    lmb, R = la.eig(L_out)
    # R is already computed in extract_out_modes, is it faster to keep
    # it in memory?
    Phi = Q @ R

    N = Phi.conj().T @ Phi

    for i, l1 in enumerate(lmb):
        for j, l2 in enumerate(lmb):
            ll = np.conj(l1) * l2
            N[i, j] *= ll / (1 - ll)

    for i, psi in enumerate(psi_alpha_0.T):  # loop if degeneracy
        a = q_e[:, i]
        norm = psi.conj().T @ psi
        norm += a.conj().T @ N @ a
        norm = np.sqrt(norm)
        psi_alpha_0[:, i] /= norm
        q_e[:, i] /= norm

    if schur_modes:
        return psi_alpha_0, q_e, L_out, Q
    else:
        return psi_alpha_0, R @ q_e, np.diag(lmb), Phi


def max_norm(X):
    """
    Norm every column of X such that the maximum value of the column is equal
    to 1
    """
    X_tilde = np.zeros(X.shape, dtype=complex)
    arg_max, max_x = [], []

    for i, x in enumerate(X.T):
        # take the maximum to avoid division by small numbers
        x_max = np.argmax(abs(x))
        X_tilde[:, i] = x / x[x_max]
        arg_max.append(x_max)
        max_x.append(x[x_max])
    return X_tilde, arg_max, max_x
