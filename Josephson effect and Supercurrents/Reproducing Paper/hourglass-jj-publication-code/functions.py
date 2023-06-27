"""
Copyright (c) 2018 and later, Muhammad Irfan and Anton Akhmerov.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
"""

from functools import partial

import kwant
from kwant.digest import uniform
import numpy as np
import scipy.linalg as la
from scipy import integrate


def make_system(a, W, L, dL, Wb):
    """Make a normal scattering region (2DEG) in the shape of an hourglass
    attached to two translationally invariant leads.

    Parameters
    ----------
    a : integer
        Lattice constant of a square lattice.
    W : integer
        Width of the leads (along the y-axis) attached to the scattering region.
    L : integer
        Length of the scattering region along the x-axis between the leads.
    dL : integer
        Position of the bottleneck along the x-axis. A zero value means
        a symmetric device.
    Wb : integer
        Width of the bottleneck region along the y-axis.
    """
    def hourglass_shape(pos):
        (x, y) = pos
        left_side = 0.5 * W / (- L / 2 - dL) * (
            x - dL) + 0.5 * Wb / (dL + L / 2) * (x + L / 2)
        right_side = 0.5 * W / (L / 2 - dL) * (
            x - dL) + 0.5 * Wb / (-dL + L / 2) * (L / 2 - x)
        if -L / 2 <= x <= dL:
            return (x >= -L / 2
                    and x <= L / 2
                    and abs(y) <= np.round(left_side))
        elif dL < x <= L / 2:
            return (x >= -L / 2
                    and x <= L / 2
                    and abs(y) <= np.round(right_side))

    def onsite(site, par):
        (x, y) = site.pos
        disorder = 0.5 * par.U0 * ((2 * uniform(repr(site), salt='')) - 1)
        return 4 * par.t - par.mu - disorder

    def hopx(site1, site2, par):
        xt, yt = site1.pos
        xs, ys = site2.pos
        phase = np.exp(-0.5 * np.pi * 1j * par.flux * (xt - xs) * (yt + ys))
        return -par.t * phase

    def hopy(site1, site2, par):
        return -par.t

    lat = kwant.lattice.square(a, norbs=1)
    syst = kwant.Builder()
    syst[lat.shape(hourglass_shape, (0, 0))] = onsite
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)), time_reversal=1)

    def lead_shape(pos):
        (x, y) = pos
        return -W / 2 <= y <= W / 2

    def lead_hopx(site1, site2, par):
        return -par.t

    def lead_onsite(site, par):
        return 4 * par.t - par.mu

    lead[lat.shape(lead_shape, (-1, 0))] = lead_onsite
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = lead_hopx
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    syst = syst.finalized()
    return syst


def supercurrent_tight_binding(smatrix, phi, Delta):
    """Returns the supercurrent in a SNS Josephson junction using
    a tight-binding model.

    Parameters
    ----------
    smatrix : kwant.smatrix object
        Contains scattering matrix and information of lead modes.
    phi : float
        Superconducting phase difference between two superconducting leads.
    Delta : float
        Superconducting gap.
    """
    N, M = [len(li.momenta) // 2 for li in smatrix.lead_info]
    s = smatrix.data
    r_a11 = 1j * np.eye(N)
    r_a12 = np.zeros((N, M))
    r_a21 = r_a12.T
    r_a22 = 1j * np.exp(- 1j * phi) * np.eye(M)
    r_a = np.bmat([[r_a11, r_a12], [r_a21, r_a22]])
    # Matrix
    A = (r_a.dot(s) + (s.T).dot(r_a)) / 2
    # dr_a/dphi
    dr_a11 = np.zeros((N, N))
    dr_a12 = np.zeros((N, M))
    dr_a21 = dr_a12.T
    dr_a22 = np.exp(-1j * phi) * np.eye(M)
    dr_a = np.bmat([[dr_a11, dr_a12], [dr_a21, dr_a22]])
    # dA/dphi
    dA = (dr_a.dot(s) + (s.T).dot(dr_a)) / 2
    # d(A^dagger*A)/dphi
    Derivative = (dA.T.conj()).dot(A) + (A.T.conj()).dot(dA)
    Derivative = np.array(Derivative)
    eigVl, eigVc = la.eigh(A.T.conj().dot(A))
    eigVl = Delta * eigVl ** 0.5
    eigVc = eigVc.T
    current = np.sum((eigVc.T.conj().dot(Derivative.dot(eigVc)) / eigVl)
                     for eigVl, eigVc in zip(eigVl, eigVc))
    current = 0.5 * Delta ** 2 * current.real
    return current

# ============================================================================
# Supercurrent density maps
# ============================================================================


def andreev_states(syst, par, phi, Delta):
    """Returns Andreev eigenvalues and eigenvectors.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem object
        A finalized kwant system having a scattering region
        connected with two semi-infinite leads.
    par : SimpleNamespace object
        Simplenamespace object with Hamiltonian parameters.
    phi : float
        Superconducting phase difference between the two superconducting leads.
    Delta : float
        Superconducting gap.
    """
    s = kwant.smatrix(syst, energy=0, args=[par])
    N, M = [len(li.momenta) // 2 for li in s.lead_info]
    s = s.data
    r_a11 = 1j * np.eye(N)
    r_a12 = np.zeros((N, M))
    r_a21 = r_a12.T
    r_a22 = 1j * np.exp(- 1j * phi) * np.eye(M)
    r_a = np.bmat([[r_a11, r_a12], [r_a21, r_a22]])
    zeros = np.zeros(shape=(len(s), len(s)))
    matrix = np.bmat([[zeros, (s.T.conj()).dot(r_a.conj())],
                      [(s.T).dot(r_a), zeros]])
    eigVl, eigVc = la.eig(matrix)
    eigVc = la.qr(eigVc)[0]
    eigVl = eigVl * Delta
    values = []
    vectors = []
    for ii in range(len(eigVl)):
        if eigVl[ii].real > 0 and eigVl[ii].imag > 0:
            values.append(eigVl[ii].real)
            vectors.append(eigVc.T[ii][0:len(eigVl) // 2])
    values = np.array(values)
    vectors = np.array(vectors)
    return values, vectors


def andreev_wf(eigvec, kwant_wf):
    """
    Returns Andreev wavefunctions using eigenvalues and eigenvectors from
    the bound-state eigenvalue problem.

    Parameters
    ----------
    eigvec : numpy array
        Eigenvectors from the Andreev bound-state condition.
    kwant_wf : kwant.solvers.common.WaveFunction object
        Wavefunctions of a normal scattering region connected
        with two normal leads.
    """
    w = np.vstack((kwant_wf(0), kwant_wf(1)))
    and_wf = [np.dot(vec, w) for vec in eigvec]
    return and_wf


def intensity(syst, psi, par):
    """Returns the current through a kwant system.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem object
        A finalized kwant system having a scattering region connected
        with two semi-infinite leads.
    psi : numpy array
        Andreev wavefunctions constructed from kwant wavefunctions and
        Andreev bound-state eigenvalue problem.
    par : SimpleNamespace object
        Simplenamespace object with Hamiltonian parameters.
    """
    I_operator = kwant.operator.Current(syst)
    return sum(I_operator(psi_i, args=[par]) for psi_i in psi)

# ============================================================================
#  Quasiclassical calculations for asymmetric hourglass device
# ============================================================================


def bounds_theta_asymmetric(x, par):
    """Returns the angle bounds for allowed trajectories.

    Parameters
    ----------
    x : float
        The starting point of the trajectory from a superconducting lead at -L/2.
    par : SimpleNamespace object
        Container with parameters defining the geometry of the device.
    """
    _min, _max = [((s*par.Wb / 2 - x) / (par.L / 2 - par.dL),
                   (s*par.W / 2 - x) / par.L) for s in [-1, 1]]
    return [np.arctan(x) for x in [max(_min), min(_max)]]


def bounds_x_asymmetric(par):
    """Returns the effective width of an asymmetric hourglass device
    which gives reflectionless trajectories.

    Parameters
    ----------
    par : SimpleNamespace object
        Container with parameters defining the geometry of the device.
    """
    if par.dL > par.Wb:
        x0 = ((par.L / 2 - par.dL)
              * (par.W / 2 + par.Wb / 2)
              / (par.L / 2 + par.dL) + par.Wb / 2)
        return [x0, -x0]
    else:
        return [-par.W / 2, par.W / 2]


def quasiclassical_supercurrent_asymmetric(par):
    """Compute supercurrent in a symmetric or asymmetric
    hourglass SNS Josephson junction.

    Parameters
    ----------
    par : SimpleNamespace object
        Object with parameters about the geometry of
        the device as well as the applied magnetic field
        and superconducting phase difference.
    """
    def f(theta, x):
        gamma = 2 * par.L / par.lm ** 2 * (x + 0.5 * par.L * np.tan(theta))
        return np.cos(theta) * np.sin(par.phi - gamma)
    x_bounds = partial(bounds_x_asymmetric, par=par)
    theta_bounds = partial(bounds_theta_asymmetric, par=par)
    return integrate.nquad(f, [theta_bounds, x_bounds], opts={'limit': 500})

# ==============================================================================
# Quasiclassical calculations for tunable densities
# ==============================================================================


def trajectory_length(x, theta, par):
    """Returns projection of trajectory-length on x-axis for a trajectory
    starting at point x from the lead at -L/2 and ending at the other
    lead at L/2.

    Parameters
    ----------
    x : float
        The starting position of a trajectory from the
        superconducting lead at -L/2
    theta : float
        Maximum or minimum angle a trajectory starting at point x
        can make to reach the other side of bottle-neck.
    par : SimpleNamespace object
        Simplenamespace object with parameters defining the geometry of the
        device and Fermi wavevectors on both sides of the bottleneck.
    """
    # k2 > k1 Necessary condition
    angle = np.arcsin(np.sin(theta) * par.k1 / par.k2)
    val = x + 0.5 * par.L * (np.tan(theta) + np.tan(angle))
    return val


def lower_angle_bounds(x, par):
    """Returns angle bounds for trajectories starting at -L/2 to enter
    the other side of bottleneck region.

    Parameters
    ----------
    x : float
        The starting position of a trajectory from the
        superconducting lead at -L/2.
    par : SimpleNamespace object
        Container with parameters about the geometry of the device.
    """
    return [np.arctan(2 * (s * par.Wb / 2 - x) / par.L) for s in [-1, 1]]


def theta_bounds(x, par):
    """Returns the angle bounds on trajectories starting at -L/2 to
    reach the superconducting lead at L/2 without
    any edge reflection.

    Parameters
    ----------
    x : float
        The starting position of a trajectory from the
        superconducting lead at -L/2.
    par : SimpleNamespace object
        Simplenamespace object with parameters defining the geometry of
        the device and Fermi wavevectors on both sides
        of the bottleneck.
    """
    min_theta, max_theta = lower_angle_bounds(x, par)
    if x < 0:
        thetas = np.linspace(max_theta, min_theta, 10000)
        for theta in thetas:
            length = trajectory_length(x, theta, par)
            if abs(length) < par.W / 2:
                return [min_theta, theta]
    else:
        thetas = np.linspace(min_theta, max_theta, 10000)
        for theta in thetas:
            length = trajectory_length(x, theta, par)
            if abs(length) < par.W / 2:
                return [theta, max_theta]


def bounds_x(par):
    """Returns the bounds on the width of the superconducting leads.

    Parameters
    ----------
    par : SimpleNamespace object
        Container with parameters defining the geometry of the device.
    """
    return [-par.W / 2, par.W / 2]


def quasiclassical_supercurrent_tunable_densities(par):
    """Returns supercurrent for a given set of parameters for
    symmetric hourglass device with unequal carrier
    densities on both sides of the bottleneck.

    Parameters
    ----------
    par: SimpleNamespace object
         Container with device geomtry and Hamiltonian parameters.
    """
    def f(theta, x):
        theta_prime = np.arcsin(np.sin(theta) * par.k1 / par.k2)
        gamma = 2 * par.L / par.lm ** 2 * (x + par.L * (
            3 * np.tan(theta) + np.tan(theta_prime)) / 8)
        res = np.cos(theta) * np.sin(par.phi - gamma)
        return res
    xbounds = partial(bounds_x, par=par)
    thetabounds = partial(theta_bounds, par=par)
    return integrate.nquad(f, [thetabounds, xbounds], opts={'limit': 500})
