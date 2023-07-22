"""
This library is used to find and compute boundstat wave function.

'standard' module use SVD decompostion.

'fast' module use eigenvalue decomposition.
It is faster that SVD but may return false positive.
"""

from . import fast, standard
