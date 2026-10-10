
"""
SISO flat Rayleigh fading channel.
Inputs:
    Data  : nonempty 1D NumPy array of complex symbols
    Nzero : nonnegative finite noise variance
Outputs:
    received : complex received symbols
    h        : complex channel coefficients
    """

import numpy as np


def Rayleigh_Channel(Data, Nzero):

    Data = np.asarray(Data)

    if Data.ndim != 1 or Data.size == 0:
        raise ValueError("Data must be a nonempty 1D array")

    if not np.all(np.isfinite(Data)):
        raise ValueError("Data must contain finite values")

    if not np.isscalar(Nzero) or not np.isrealobj(Nzero):
        raise ValueError("Nzero must be a real scalar")

    if not np.isfinite(Nzero) or Nzero < 0:
        raise ValueError("Nzero must be finite and non-negative")

    N = Data.size

    # Generate complex Gaussian fading coefficients
    h = (
        np.random.randn(N) + 1j * np.random.randn(N)
    ) / np.sqrt(2)

    # Generate complex AWGN
    noise = np.sqrt(Nzero / 2) * (
        np.random.randn(N) + 1j * np.random.randn(N)
    )

    # Apply fading and additive noise
    received = h * Data + noise

    return received, h
