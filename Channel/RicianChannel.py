"""
@author: razanalkakoun
SISO flat Rician fading channel.

Inputs:
    Data  : nonempty 1D NumPy array of complex symbols
    Nzero : nonnegative finite noise variance
    K     : nonnegative Rician K-factor (linear scale)

Outputs:
    received : complex received symbols
    h        : complex channel coefficients

Model:
    y = h*x + n

    h = sqrt(K/(K+1)) + sqrt(1/(K+1))*g
    g ~ CN(0,1)
    n ~ CN(0,Nzero)

Each symbol experiences independent Rician fading.
"""

import numpy as np


def Rician_Channel(Data, Nzero, K=1.0):

    Data = np.asarray(Data)

    if Data.ndim != 1 or Data.size == 0:
        raise ValueError("Data must be a nonempty 1D array")

    if not np.all(np.isfinite(Data)):
        raise ValueError("Data must contain finite values")

    if not np.isscalar(Nzero) or not np.isrealobj(Nzero):
        raise ValueError("Nzero must be a real scalar")

    if not np.isfinite(Nzero) or Nzero < 0:
        raise ValueError("Nzero must be finite and non-negative")

    if not np.isscalar(K) or not np.isrealobj(K):
        raise ValueError("K must be a real scalar")

    if not np.isfinite(K) or K < 0:
        raise ValueError("K must be finite and non-negative")

    N = Data.size

    # Generate scattered complex Gaussian component
    g = (
        np.random.randn(N) + 1j * np.random.randn(N)
    ) / np.sqrt(2)

    # Rician fading coefficient
    h = (
        np.sqrt(K / (K + 1))
        + np.sqrt(1 / (K + 1)) * g
    )

    # Generate complex AWGN
    noise = np.sqrt(Nzero / 2) * (
        np.random.randn(N) + 1j * np.random.randn(N)
    )

    # Apply fading and noise
    received = h * Data + noise

    return received, h
