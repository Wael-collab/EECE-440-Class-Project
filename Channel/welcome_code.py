"""
EECE 640 - Wireless Communication, Class Project
MODULE: CHANNEL

Purpose
-------
Simulates propagation and noise. Every model is a special case of the
discrete-time baseband model

    y[m] = sum_{l=0}^{L-1} h_l[m] x[m-l] + w[m],     w[m] ~ CN(0, N0),

and differs in the number of taps L (flat or frequency-selective) and in how
the taps vary in time (new realization every sample, or one per block).

Files
-----
AWGNChannel.py            AWGN_Channel(Data, N0) -> y
                          Additive white Gaussian noise, any array shape.
Rayleigh.py               Rayleigh_Channel(Data, N0) -> (y, h)
                          SISO flat Rayleigh fading, independent per sample.
RicianChannel.py          Rician_Channel(Data, N0, K) -> (y, h)
                          SISO flat Rician fading, independent per sample.
MIMORicianFlatChannel.py  MIMORicianFlatChannel(X, N0, Nt, Nr, K, H_los) -> (Y, H)
                          MIMO flat block fading; K = 0 gives i.i.d. Rayleigh.
MIMOLTapChannel.py        MIMOLTapChannel(L, X, N0, Nt, Nr) -> (Y, h)
                          MIMO L-tap Rayleigh channel, constant per block;
                          output has T + L - 1 samples.
OFDMChannel.py            OFDM(symbols, L, N0) -> (y, h)
                          Legacy self-contained SISO OFDM link
                          (IDFT, cyclic prefix, L-tap channel, DFT).
AWSN.py                   Alpha_Stable_Channel(Data, alpha, gamma) -> y
                          Symmetric alpha-stable impulsive noise (needs the
                          aub_htp package); alpha = 2 is Gaussian, N0 = 4 gamma^2.

Conventions shared with the Transmitter and Receiver modules
------------------------------------------------------------
* E[sum_l |h_l|^2] = 1 for every model, so the average received SNR equals
  the transmitted Es/N0.
* Noise is circularly symmetric: Re{w}, Im{w} ~ N(0, N0/2), independent.
* Channel coefficients are returned so the Receiver can use them as perfect CSI.
* MIMO shapes: X is (Nt, T), Y is (Nr, T), H is (Nr, Nt) or (Nr, Nt, L).

Usage
-----
    from Channel.MIMORicianFlatChannel import MIMORicianFlatChannel
    from Channel.MIMOLTapChannel import MIMOLTapChannel

    Y, H = MIMORicianFlatChannel(X, N0, Nt=2, Nr=2, K=0)    # flat block fading
    Y, h = MIMOLTapChannel(4, X, N0, Nt=2, Nr=2)            # 4-tap multipath

Known limitations
-----------------
* No time-correlated (Doppler) fading: coherence time is one sample (SISO
  models) or one block (MIMO models).
* The default H_los of MIMORicianFlatChannel is the all-ones (rank-one) matrix;
  pass a full-rank LoS matrix for spatial multiplexing at large K.
* Uniform power-delay profile only; Alpha_Stable_Channel accepts 1-D input only.
* The modules use NumPy's global random generator; seed it for reproducibility.

Run this file for a short demonstration:
    python Channel/welcome_code.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo():
    from Channel.Rayleigh import Rayleigh_Channel
    from Channel.RicianChannel import Rician_Channel
    from Channel.MIMORicianFlatChannel import MIMORicianFlatChannel
    from Channel.MIMOLTapChannel import MIMOLTapChannel

    np.random.seed(0)
    x = np.ones(100_000, dtype=complex)
    _, h = Rayleigh_Channel(x, 0.0)
    print(f"Rayleigh:        E|h|^2 = {np.mean(np.abs(h) ** 2):.3f}")
    _, h = Rician_Channel(x, 0.0, 10)
    print(f"Rician (K = 10): E|h|^2 = {np.mean(np.abs(h) ** 2):.3f}, "
          f"|E h| = {abs(np.mean(h)):.3f}")

    X = np.ones((2, 100), dtype=complex)
    Y, H = MIMORicianFlatChannel(X, 0.1, 2, 2, K=0)
    print(f"MIMO flat 2x2:   X {X.shape} -> Y {Y.shape}, H {H.shape}")
    Y, h = MIMOLTapChannel(4, X, 0.1, 2, 2)
    print(f"MIMO 4-tap 2x2:  X {X.shape} -> Y {Y.shape} (T + L - 1), h {h.shape}")


if __name__ == "__main__":
    print("Channel module - EECE 640 Wireless Communication class project")
    demo()
