"""
Compute bound state using eigenvalue.

Use 'scan_eig_values' to have a guest of energies of bound states.
Bound stats are located where singular value = 0.

Use 'finder' to find the exact energy of bound states in a small
range of energies.

Use 'solver' to obtain the wave-function of the bound states at a
precise energy.
'solver' may return false positive.
"""

from __future__ import division
import numpy as np
import scipy.linalg as la
from scipy.optimize import newton
np.set_printoptions(precision=16, suppress=1)
import scipy.sparse as sp
from ._common import *
from .standard import _singular_values


def solver(syst, E, args=(), eps=1e-4, sparse=False):
    """
    Raise an error if there is no bound state (singular value > eps)
    Parameters:
    transf: numpy array or sparse csr matrix. V.dot(transf) gives the
    hopping matrix between the scattering region and the leads.
    Returns:
    The normalized wavefunction of the bound state,
    psi_alpha_0: wavefunction in the scattering region
    q_e: numpy array, as defined in Eq.7
    L_out: Evanescent eigenvalues
    Q: part of the wavefunction in the lead.

    Notes:
    This function will be useless once the BS_solver will be included in kwant
    """
    H_s, H_leads, V_leads, transf = extract_kwant_matrices(syst, args=args,
                                                           sparse=sparse)

    return _solver(H_s, H_leads, V_leads, transf, E, eps, sparse)


def _solver(H_s, H_leads, V_leads, transf, E, eps=1e-4, sparse=False):
    """
    Returns the elements to compute the wavefunction of a bound state
    Raise an error if there is no bound state (singular value > eps)
    Parameters:
    transf: numpy array or sparse csr matrix. V.dot(transf) gives the
    hopping matrix between the scattering region and the leads.
    Returns:
    The normalized wavefunction of the bound state,
    psi_alpha_0: wavefunction in the scattering region
    q_e: numpy array, as defined in Eq.7
    L_out: Evanescent eigenvalues
    Q: part of the wavefunction in the lead.

    Notes:
    This function will be useless once the BS_solver will be included in kwant
    """

    vals, vecs, L_leads, X_out_leads = _eig_values(H_s, H_leads, V_leads,
                                                   transf, E, sparse=sparse,
                                                   uv=True)
    N = H_s.shape[0]
    return compute_wf(vals, vecs, L_leads, X_out_leads, N, eps=eps)


def basic_solver(H_0, V, E):
    """
    Setup left and right hand side matrices from general eigenproblem
    in Eq.5.

    returns:
    A, B: numpy arrays, left and right hand sides
    X_full: numpy array, matrix that countains the eigenvectors from Eq.5
    lmb_eva: eigenvalues of the system A X_full = lmb B X_full
    """
    zero = np.zeros(H_0.shape, dtype=complex)
    Id = np.eye(H_0.shape[0], dtype=complex)

    A = np.hstack((H_0 - E*Id, V.conj().T))
    A = np.vstack((A, np.hstack((Id, zero))))

    B = np.hstack((-V, zero))
    B = np.vstack((B, np.hstack((zero, Id))))

    # need_orth is set to False. If X_eva is orthogonal, lmb is not diagonal
    # which makes the code fail.
    lmb_eva, X_eva = extract_out_modes(H_0, V, E, return_mat=False,
                                       need_orth=False)

    X_full = np.zeros(shape=(A.shape[0], lmb_eva.shape[0]), dtype=complex)
    for i, (l, phi) in enumerate(zip(np.diag(lmb_eva), X_eva.T)):
        X_full[:, i] = np.hstack((phi, l * phi))

    return lmb_eva, X_full, (A, B)


def direct_method(A, B, A_dot, lmb, X):
    """
    Compute the derivative of the eigenvectors X and eigenvalues
    lmb
    It doesn't need the left eigenvectors, so it is a direct method like
    described in Ref.28 (doi=10.1002/nme.1620260202)
    The eigenproblem is A X = lmb B X.

    Returns:
    lmb_dot: the derivatives of the eigenvalues
    X_dot: derivative of the eigenvectors (the full one).

    Notes:
    Probably not the fastest as it has to solve a linear system
    for each eigenvector. Maybe an adjoint method would be better?
    """
    X, arg_max, max_x = max_norm(X)

    N = A.shape[0] / 2

    lmb_dot = []
    X_dot = np.zeros(shape=X.shape, dtype=complex)
    for i, (l, x) in enumerate(zip(lmb, X.T)):
        lhs = np.hstack(((A - l * B), -np.dot(B, x)[:, None]))
        # remove column m because of norm
        lhs = np.delete(lhs, arg_max[i], axis=1)
        rhs = -np.dot(A_dot, x)
        sol = la.solve(lhs, rhs)

        lmb_dot.append(sol[-1])
        x_dot = sol[:-1]
        x_dot *= max_x[i]
        x_dot = x_dot.tolist()
        x_dot.insert(arg_max[i], 0)  # put back column m
        X_dot[:, i] = x_dot  # go back to initial norm

    return np.asarray(lmb_dot), X_dot


def derivative_lhs(E, H_s, H_leads, V_leads, transf, sparse=False):
    """
    Compute the derivative of the lhs of Eq.(14) (H_eff in its hermitian
    formulation) wrt energy.
    """

    X, X_dot = [], []
    L, L_dot = [], []

    for H_l, V_l in zip(H_leads, V_leads):

        lmb, X_full, (A, B) = basic_solver(H_l, V_l, E)

        N_l = V_l.shape[0]
        A_dot = np.zeros(shape=A.shape, dtype=complex)
        A_dot[:N_l, :N_l] = -np.eye(N_l)

        lmb_inv = 1 / np.diag(lmb)
        lmb_dot_inv, X_dot_full = direct_method(A, B, A_dot, lmb_inv, X_full)
        # because original problem formulated in 1/lmb
        lmb_dot = -1 / lmb_inv**2 * lmb_dot_inv

        X_dot.append(X_dot_full[:N_l, :])
        X.append(X_full[:N_l, :])
        L.append(lmb)
        L_dot.append(np.diag(lmb_dot))

    # unpack and make block diagonal matrices for every lead
    H, V = block_diag(*H_leads), block_diag(*V_leads)
    X, L = block_diag(*X), block_diag(*L)
    X_dot, L_dot = block_diag(*X_dot), block_diag(*L_dot)

    N_s = H_s.shape[0]
    N_eva = X.shape[1]
    sh = (N_s + N_eva, N_s + N_eva)
    LX = X_dot.dot(L) + X.dot(L_dot)

    if sparse:
        dot = sp.csr_matrix.dot
        V = sp.csr_matrix(V)
        L, X = sp.csr_matrix(L), sp.csr_matrix(X)
        X_dot, LX = sp.csr_matrix(X_dot), sp.csr_matrix(LX)

        V_p = dot(V, transf)
        top = sp.hstack((-sp.eye(N_s), dot(V_p.conj().T.tocsr(), LX)))

        LX_dag = LX.conj().T.tocsr()
        bottom_left = dot(LX_dag, V_p)
        bottom_right = -dot(LX_dag, dot(V, X))
        bottom_right -= dot(dot(dot(L.conj(), X.conj().T), V), X_dot)
        bottom = sp.hstack((bottom_left, bottom_right))
        H_prime = sp.vstack((top, bottom))
    else:
        H_prime = np.zeros(shape=sh, dtype=complex)
        H_prime[:N_s, :N_s] = -np.eye(N_s)
        H_prime[:N_s, N_s:] = ((V.dot(transf)).conj().T).dot(LX)
        H_prime[N_s:, :N_s] = LX.conj().T.dot(V.dot(transf))
        H_prime[N_s:, N_s:] = -LX.conj().T.dot(V.dot(X))
        H_prime[N_s:, N_s:] -= L.conj().dot(X.conj().T).dot(V).dot(X_dot)
    return H_prime


def eig_val_derivative(E, H_s, H_leads, V_leads, transf, sparse=False,
                       uv=True, need_orth=True, eig_val=2):
    """
    Compute the derivative of the smallest eigenvalue of H_eff wrt to the
    energy, Eq.20., de_alpha / dE = psi_dag dH_eff / dE psi.
    """
    if not np.isscalar(E):  # because fmin_tnc uses array and not scalars
        E = E[0]

    L_out_leads, X_out_leads = leads_modes(H_leads, V_leads, E,
                                           need_orth=False)
    H = setup_lhs_H(H_s, V_leads, transf, L_out_leads, X_out_leads, E,
                    sparse=sparse)
    # if eig_val_derivative is called from a root finder, then setup_lhs_H()
    # has already been called in eig_values() at the same energy.
    # Should be fixed...

    if sparse:
        vals, vecs = sp.linalg.eigsh(H, k=3, sigma=0)
        min_val = np.argmin(abs(vals))  # find the closest eig value to 0
        vec = vecs[:, min_val]
        H_prime = derivative_lhs(E, H_s, H_leads, V_leads,
                                 transf, sparse=sparse)

        return vec.conj().T.dot(H_prime.todense()).dot(vec)[0, 0]
    else:
        # compute all eigenvalues
        vals, vecs = la.eigh(H)
        min_val = np.argmin(abs(vals))  # find the closest eig value to 0
        vec = vecs[:, min_val]
        # ~ vec /= np.sqrt(np.dot(vec.conj().T, vec)) # already normalized to 1

        H_prime = derivative_lhs(E, H_s, H_leads, V_leads,
                                 transf, sparse=sparse)
        # Simple expression because of hermitian formulation
        return vec.conj().T.dot(H_prime).dot(vec)


def eig_values(syst, E, args=(), sparse=False, uv=True,
               need_orth=True, sing_values=4, tol=1e-5):
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
    eig_val: the number of eigenvalues to be computed. Only if sparse
             is True
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
    return _eig_values(H_s, H_leads, V_leads, transf, E, sparse, uv,
                       need_orth, sing_values, tol)


def _eig_values(H_s, H_leads, V_leads, transf, E, sparse=False, uv=True,
                need_orth=True, sing_values=4, tol=1e-5):
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
    eig_val: the number of eigenvalues to be computed. Only if sparse
             is True
    returns:
    if uv is false: the singular values of the matrix L
    else:
    S: The singular values of L
    Wh: The right matrix of the svd decomposition
    L: array that countains the evanescent outgoing lambdas
    X_out: Columns are the corresponding evanescent wavefunctions
    """

    L_out_leads, X_out_leads = leads_modes(H_leads, V_leads, E, tol=tol)

    lhs = setup_lhs_H(H_s, V_leads, transf, L_out_leads, X_out_leads, E,
                      sparse=sparse)

    if sparse:
        if not uv:
            evals = sp.linalg.eigsh(lhs, return_eigenvectors=False,
                                    k=eig_val, sigma=0, which='LM')
            return evals
        else:
            evals, evecs = sp.linalg.eigsh(lhs, return_eigenvectors=True,
                                           k=eig_val, sigma=0, which='LM')
        return evals, evecs, L_out_leads, X_out_leads
    else:
        if not uv:
            return la.eigh(lhs, eigvals_only=True)
        else:
            vals, vecs = la.eigh(lhs)
            return vals, vecs, L_out_leads, X_out_leads


def setup_lhs_H(H_s, V_leads, transf, L_out_leads, X_out_leads, E,
                sparse=False):
    """
    Setup the left hand side of the equation to find bound states.
    Parameters:
    H_s: numpy matrix describing the scattering region
    V_leads: sequence of numpy arrays describing the inter cell hopping
             in the leads
    X_out_leads: sequence of array of eigenmodes in the leads
    E: energy, float
    """

    L_out = block_diag(*L_out_leads)
    X_out = block_diag(*X_out_leads)
    V = block_diag(*V_leads)

    if sparse:  # Uses either scipy.sparse if sparse of numpy if dense
        d = sp.csr_matrix.dot
        lin = sp
        V = sp.csr_matrix(V)
        L_out, X_out = sp.csr_matrix(L_out), sp.csr_matrix(X_out)
    else:
        d = np.dot
        lin = np

    Id_N = lin.eye(H_s.shape[0])
    V_ls = d(V, transf)
    X_L = d(X_out, L_out)

    if sparse:
        L_X = X_L.conj().T.tocsr()
        lhs = lin.hstack([H_s - E * Id_N, d(V_ls.conj().T.tocsr(), X_L)])
    else:
        L_X = X_L.conj().T

        lhs = lin.hstack([H_s - E * Id_N, d(V_ls.conj().T, X_L)])

    lhs = lin.vstack([lhs, lin.hstack([d(L_X, V_ls), -d(L_X, d(V, X_out))])])
    return lhs


def finder(syst, E_0, args=(), sparse=False, tol=1e-6, sing_values=4,
           uv=False, fprime=eig_val_derivative):
    """
    Function that search for a minimum of the eigenvalues of the lhs of Eq.13
    Uses the newton algorithm.

    Parameters:
    E_0 : float. An initial estimate of the bound stat energy.
    fprime: None or function (same arguments for eig_val and
            eig_val_derivative) that returns the derivative of an
            eigenvalue of Eq.13

    Returns:
    E: The energy of the bound state if it found one or print a message

    Important note:
    If there are degenerated eigenvalues, then fprime should be equal to
    None, as I have no clue how to compute the derivative of fully degenerated
    eigenvalues (or the derivative of an invariant subspace).
    Ideas welcome...
    """
    H_s, H_leads, V_leads, transf = extract_kwant_matrices(syst, args=args,
                                                           sparse=sparse)
    return _finder(H_s, H_leads, V_leads, transf, E_0, sparse,
                   tol, sing_values, uv, fprime)


def _finder(H_s, H_leads, V_leads, transf, E_0, sparse=False,
            tol=1e-6, sing_values=4, uv=False, fprime=eig_val_derivative):
    """
    Function that search for a minimum of the eigenvalues of the lhs of Eq.13
    Uses the newton algorithm.

    Parameters:
    fprime: None or function (same arguments for eig_val and
            eig_val_derivative) that returns the derivative of an
            eigenvalue of Eq.13

    Returns:
    E: The energy of the bound state if it found one or print a message

    Important note:
    If there are degenerated eigenvalues, then fprime should be equal to
    None, as I have no clue how to compute the derivative of fully degenerated
    eigenvalues (or the derivative of an invariant subspace).
    Ideas welcome...
    """

    def dumb(e, H_s, H_leads, V_leads, transf, sparse=sparse):
        eig = _eig_values(H_s, H_leads, V_leads, transf, e, sparse=sparse,
                          uv=False, sing_values=sing_values)
        idx = np.argmin(abs(eig))
        return eig[idx]
    try:
        E = newton(dumb, E_0, fprime=fprime, tol=tol,
                   args=(H_s, H_leads, V_leads, transf, sparse))

        if false_positive_BS(H_s, H_leads, V_leads, transf, E, eps=tol):
            print('False positive, no bound state')
            return False
        else:

            return E
    except RuntimeError:
        print('no bound states found')


def scan_eig_values(syst, e_min, e_max, args=(), N=500, sparse=False, k=5):
    energies = np.linspace(e_min, e_max, N)
    """Compute the eigenvalues on a energy range.
    See 'eig_values' for details.
    """
    H_s, H_leads, V_leads, transf = extract_kwant_matrices(syst, args=args,
                                                           sparse=sparse)
    return _scan_eig_values(H_s, H_leads, V_leads, transf, e_min, e_max, N,
                            sparse, k)


def _scan_eig_values(H_s, H_leads, V_leads, transf, e_min, e_max, N=500,
                     sparse=False, k=5):
    energies = np.linspace(e_min, e_max, N)
    """Compute the eigenvalues on a energy range.
    See 'eig_values' for details.
    """
    if sparse:
        dumb = np.zeros(shape=(N, k))
    else:
        # at maximum the number of singular value is the dimension of H_s
        # + the dimension of all the leads together
        dumb = np.zeros(shape=(N, H_s.shape[0] + sum(V.shape[0]
                                                     for V in V_leads)))
        # the zeros are later replaced with NaN so they are not plotted.

    for i, e in enumerate(energies):
        s = _eig_values(H_s, H_leads, V_leads, transf, e, sparse,
                        uv=False, sing_values=k)

        dumb[i, :len(s)] = s
    dumb = fill_zero_with_nan(dumb)

    return (energies, dumb)


def false_positive_BS(H_s, H_leads, V_leads, transf, E, eps=1e-8,
                      sparse=False):
    """
    Check the False positive computing the singular values of H_eff in
    it non-hermitian form.
    It would be also possible to check Eq.19, something like:
    np.amax(block_diag(V_leads) @ (P @ psi_alpha_0 - phi_e @ q_e)) < eps
    """
    sing = _singular_values(H_s, H_leads, V_leads, transf, E, uv=False,
                            sparse=False)
    if np.amin(sing) < eps:
        return False  # True bound state
    else:
        # True means a true "false positive" (i.e., no bound states)
        return True
