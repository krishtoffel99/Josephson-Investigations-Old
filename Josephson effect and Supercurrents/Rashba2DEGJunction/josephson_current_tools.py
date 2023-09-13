## COPIED OVER FROM funcs.py in data.zip (https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.119.187704) FRO TESTING PURPOSES.
import kwant 
import numpy as np
import scipy
import types

constants = types.SimpleNamespace(
    m_eff=0.015 * scipy.constants.m_e,  # effective mass in kg
    hbar=scipy.constants.hbar,
    m_e=scipy.constants.m_e,
    eV=scipy.constants.eV,
    e=scipy.constants.e,
    meV=scipy.constants.eV * 1e-3,
    k=scipy.constants.k / (scipy.constants.eV * 1e-3),
    current_unit=scipy.constants.k * scipy.constants.e / scipy.constants.hbar * 1e9,  # to get nA
    mu_B=scipy.constants.physical_constants['Bohr magneton'][0] / (scipy.constants.eV * 1e-3),
    t=scipy.constants.hbar**2 / (2 * 0.015 * scipy.constants.m_e) / (scipy.constants.eV * 1e-3 * 1e-18),
    c=1e18 / (scipy.constants.eV * 1e-3))

# Functions related to calculating the supercurrent.

def get_cuts(syst, lat, x_left=0, x_right=1):
    """Get the sites at two postions of the specified cut coordinates.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    lat : dict
        A container that is used to store Hamiltonian parameters.
    """
    l_cut = [lat(*tag) for tag in [s.tag for s in syst.sites()] if tag[0] == x_left] # <- Returns a list of all sites with x_coord = x_left
    r_cut = [lat(*tag) for tag in [s.tag for s in syst.sites()] if tag[0] == x_right] # <- Returns a list of all site with x_coord = x_left
    assert len(l_cut) == len(r_cut), "x_left and x_right use site.tag not site.pos!"
    return l_cut, r_cut


def add_vlead(syst, lat, l_cut, r_cut , parameters):
    dim = lat.norbs * (len(l_cut) + len(r_cut)) #<- (len(l_cut) + len(r_cut) gives you the number of sites in the left cut and right cut.
    # We imagine connecting virtual leads to the left and right cuts.
    # The dimensions of a cell-Hamiltonian in the left lead will be dimL = norbs*(no. of sites in the left cut). Likewise for the right lead.
    # So I guess the total Hamiltonian for the leads can be expressed in a block diagonal form? Hence the total dimension of the matrix will be 
    # dimL + dimR = dim.
    vlead = kwant.builder.SelfEnergyLead(
        lambda energy, args: np.zeros((dim, dim)), l_cut + r_cut , parameters)
    # kwant.builder.SelfEnergyLead takes 3 inputs: selfenergy_func , a sequence of site instances , parameters on which the leads depend.

    syst.leads.append(vlead)
    return syst


def hopping_between_cuts(syst, r_cut, l_cut):
    '''
    Calculates the submatrix of the Hamiltonian that acts between the sites in r_cut and l_cut
    '''
    r_cut_sites = [syst.sites.index(site) for site in r_cut]
    l_cut_sites = [syst.sites.index(site) for site in l_cut]

    def hopping(syst, params):
        return syst.hamiltonian_submatrix(params=params,
                                          to_sites=l_cut_sites,
                                          from_sites=r_cut_sites)[::2, ::2]
    return hopping


def matsubara_frequency(n, params):
    """n-th fermionic Matsubara frequency at temperature T.

    Parameters
    ----------
    n : int
        n-th Matsubara frequency

    Returns
    -------
    float
        Imaginary energy.
    """
    return (2*n + 1) * np.pi * params['k'] * params['T'] * 1j


def null_H(syst, params, n):
    """Return the Hamiltonian (inverse of the Green's function) of
    the electron part at zero phase.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    params : dict
        A container that is used to store Hamiltonian parameters.
    n : int
        n-th Matsubara frequency

    Returns
    -------
    numpy.array
        The Hamiltonian at zero energy and zero phase."""
    en = matsubara_frequency(n, params)
    # Matsubara greens function:
    # Out_lead and in_leads are the numbers of the leads where the current or wavefunction is extracted and injected respectively.
    # Looks like they've set both leads to be the left lead? Why's this reasonable? Surely current should be injected in lead-0 and 
    # extracted in lead-1?
    gf = kwant.greens_function(syst, en, out_leads=[0], in_leads=[0],
                               check_hermiticity=False, params=params)
    return np.linalg.inv(gf.data[::2, ::2])


def gf_from_H_0(H_0, t):
    """Returns the Green's function at a phase that is defined inside `t`.
    See doc-string of `current_from_H_0`.
    """
    H = np.copy(H_0)
    dim = t.shape[0]
    H[:dim, dim:] -= t.T.conj()
    H[dim:, :dim] -= t
    return np.linalg.inv(H)


def current_from_H_0(H_0_cache, H12, phase, params):
    """Uses Dyson’s equation to obtain the Hamiltonian for other
    values of `phase` without further inversions (calling `null_H`).

    Parameters
    ----------
    H_0_cache : list
        Hamiltonians at different imaginary energies.
    H12 : numpy array
        The hopping matrix between the two cross
        sections of where the SelfEnergyLead is attached.
    phase : float
        Phase at which the supercurrent is calculated.
    params : dict
        A container that is used to store Hamiltonian parameters.

    Returns
    -------
    float
        Total current of all terms in `H_0_list`.
    """
    I = sum(current_contrib_from_H_0(H_0, H12, phase, params)
            for H_0 in H_0_cache)
    return I


def I_c_fixed_n(syst, hopping, params, matsfreqs=500, N_brute=30):
    H_0_cache = [null_H(syst, params, n) for n in range(matsfreqs)]
    H12 = hopping(syst, params)
    fun = lambda phase: -current_from_H_0(H_0_cache, H12, phase, params)
    opt = scipy.optimize.brute(
        fun, ranges=[(-np.pi, np.pi)], Ns=N_brute, full_output=True)
    x0, fval, grid, Jout = opt
    return dict(phase_c=x0[0], current_c=-fval, phases=grid, currents=-Jout)


def current_contrib_from_H_0(H_0, H12, phase, params):
    """Uses Dyson’s equation to obtain the Hamiltonian for other
    values of `phase` without further inversions (calling `null_H`).

    Parameters
    ----------
    H_0 : list
        Hamiltonian at a certain imaginary energy.
    H12 : numpy array
        The hopping matrix between the two cross
        sections of where the SelfEnergyLead is attached.
    phase : float
        Phase at which the supercurrent is calculated.
    params : dict
        A container that is used to store Hamiltonian parameters.
    Returns
    -------
    float
        Current contribution of `H_0`.
    """
    t = H12 * np.exp(1j * phase)
    gf = gf_from_H_0(H_0, t - H12)
    dim = t.shape[0]
    H12G21 = t.T.conj() @ gf[dim:, :dim]
    H21G12 = t @ gf[:dim, dim:]
    return -4 * params['T'] * params['current_unit'] * (
        np.trace(H21G12) - np.trace(H12G21)).imag


def current_at_phase(syst, hopping, params, H_0_cache, phase,
                     tol=1e-2, max_frequencies=500):
    """Find the supercurrent at a phase using a list of Hamiltonians at
    different imaginary energies (Matsubara frequencies). If this list
    does not contain enough Hamiltonians to converge, it automatically
    appends them at higher Matsubara frequencies untill the contribution
    is lower than `tol`, however, it cannot exceed `max_frequencies`.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    hopping : function
        Function that returns the hopping matrix between the two cross sections
        of where the SelfEnergyLead is attached.
    params : dict
        A container that is used to store Hamiltonian parameters.
    H_0_cache : list
        Hamiltonians at different imaginary energies.
    phase : float, optional
        Phase at which the supercurrent is calculated.
    tol : float, optional
        Tolerance of the `current_at_phase` function.
    max_frequencies : int, optional
        Maximum number of Matsubara frequencies.

    Returns
    -------
    dict
        Dictionary with the critical phase, critical current, and `currents`
        evaluated at `phases`."""

    H12 = hopping(syst, params)
    I = 0
    for n in range(max_frequencies):
        if len(H_0_cache) <= n:
            H_0_cache.append(null_H(syst, params, n))
        I_contrib = current_contrib_from_H_0(H_0_cache[n], H12, phase, params)
        I += I_contrib
        if I_contrib == 0 or tol is not None and abs(I_contrib / I) < tol:
            return I
    # Did not converge within tol using max_frequencies Matsubara frequencies.
    if tol is not None:
        return np.nan
    # if tol is None, return the value after max_frequencies is reached.
    else:
        return I


def I_c(syst, hopping, params, tol=1e-2, max_frequencies=500, N_brute=30):
    """Find the critical current by optimizing the current-phase
    relation.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        The finilized kwant system.
    hopping : function
        Function that returns the hopping matrix between the two cross
        sections of where the SelfEnergyLead is attached.
    params : dict
        A container that is used to store Hamiltonian parameters.
    tol : float, optional
        Tolerance of the `current_at_phase` function.
    max_frequencies : int, optional
        Maximum number of Matsubara frequencies.
    N_brute : int, optional
        Number of points at which the CPR is evaluated in the brute
        force part of the algorithm.

    Returns
    -------
    dict
        Dictionary with the critical phase, critical current, and `currents`
        evaluated at `phases`."""
    H_0_cache = []
    func = lambda phase: -current_at_phase(syst, hopping, params, H_0_cache,
                                           phase, tol, max_frequencies)
    opt = scipy.optimize.brute(
        func, ranges=((-np.pi, np.pi),), Ns=N_brute, full_output=True)
    x0, fval, grid, Jout = opt
    return dict(phase_c=x0[0], current_c=-fval, phases=grid,
                currents=-Jout, N_freqs=len(H_0_cache))

def calculate_CPR(syst, hopping, params, phases, tol=0.01, max_frequencies=1000):
    H_0_cache = []
    I = [current_at_phase(syst, hopping, params, H_0_cache, phase, tol, max_frequencies)
                  for phase in phases]
    return I
    
