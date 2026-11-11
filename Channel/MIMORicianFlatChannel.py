
# -*- coding: utf-8 -*-
"""
MIMORicianFlatChannel.py

SISO/MIMO flat, block-constant Rician fading channel.

Inputs:
    symbols : complex array of shape (Nt, N)
    Nzero   : nonnegative complex AWGN variance
    Nt      : number of transmit antennas
    Nr      : number of receive antennas
    K       : Rician K-factor (linear scale)
    H_los   : optional deterministic LoS matrix (Nr, Nt)

Outputs:
    received : complex array of shape (Nr, N)
    H        : complex channel matrix (Nr, Nt)

Model:
    Y = H @ X + N

    H = sqrt(K/(K+1)) * H_los
        + sqrt(1/(K+1)) * H_nlos

    H_nlos entries ~ CN(0,1)
    Noise entries  ~ CN(0,Nzero)

The channel matrix H is fixed for the entire block.

Each LoS entry is assumed to have unit magnitude, so
E[|H[r,t]|^2] = 1.
"""

import numpy as np


def MIMORicianFlatChannel(
    symbols, Nzero, Nt, Nr, K=1.0, H_los=None
):
    symbols = np.asarray(symbols, dtype=complex)

    # Validate antenna counts
    if (
        not isinstance(Nt, (int, np.integer))
        or isinstance(Nt, (bool, np.bool_))
        or not isinstance(Nr, (int, np.integer))
        or isinstance(Nr, (bool, np.bool_))
        or Nt <= 0 or Nr <= 0
    ):
        raise ValueError("Nt and Nr must be positive integers")

    # Validate transmitted symbols
    if symbols.ndim != 2 or symbols.shape[0] != Nt:
        raise ValueError(
            "symbols must have shape (Nt, N)"
        )

    if symbols.shape[1] == 0:
        raise ValueError("Empty transmitted data")

    if not np.all(np.isfinite(symbols)):
        raise ValueError("symbols must contain finite values")

    # Validate noise variance and K-factor
    for value, name in [(Nzero, "Nzero"), (K, "K")]:
        if (
            not np.isscalar(value)
            or not np.isrealobj(value)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"{name} must be finite, real and nonnegative"
            )

    # Default deterministic LoS matrix
    if H_los is None:
        H_los = np.ones((Nr, Nt), dtype=complex)
    else:
        H_los = np.asarray(H_los, dtype=complex)

        if H_los.shape != (Nr, Nt):
            raise ValueError(
                "H_los must have shape (Nr, Nt)"
            )

        if not np.all(np.isfinite(H_los)):
            raise ValueError("H_los must be finite")

        if not np.allclose(np.abs(H_los), 1.0):
            raise ValueError(
                "Each H_los entry must have unit magnitude"
            )

    # Generate random scattered component
    H_nlos = (
        np.random.randn(Nr, Nt)
        + 1j * np.random.randn(Nr, Nt)
    ) / np.sqrt(2)

    # Construct normalized Rician channel
    H = (
        np.sqrt(K / (K + 1)) * H_los
        + np.sqrt(1 / (K + 1)) * H_nlos
    )

    # Apply flat MIMO fading
    faded = H @ symbols

    # Generate complex AWGN
    N = symbols.shape[1]

    noise = np.sqrt(Nzero / 2) * (
        np.random.randn(Nr, N)
        + 1j * np.random.randn(Nr, N)
    )

    received = faded + noise

    return received, H
