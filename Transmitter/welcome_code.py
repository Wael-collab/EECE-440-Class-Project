"""
EECE 640 - Wireless Communication, Class Project
MODULE: TRANSMITTER

Purpose
-------
Converts an information bit stream into the transmit matrix X of the channel
model  Y = H X + N  (rows = transmit antennas, columns = channel uses):

    bits --mapping--> symbols s --space-time code--> X_sym (Nt, T)
         --waveform, per antenna--> X (Nt, n_samples)

The space-time code acts on constellation symbols; the waveform (single carrier
or OFDM) is applied afterwards to each antenna row.

Files
-----
Transmitter.py
    Constellation   PAM, PSK, square QAM or a user-defined mapping; Gray labels;
                    normalized to average symbol energy A = Es.
    Waveform        "SC" (optionally with a cyclic prefix, i.e. SC-FDE) or
                    "OFDM" (unitary IDFT + cyclic prefix).
    Transmitter     bits -> symbols (symbols()) -> samples (run() / modulate()).

Coding_Schemes.py
    Repetition(L), Alamouti(), VBLAST(n_tx), DifferentialPSK(),
    DifferentialAlamouti()   -> encode(s) returns X_sym of shape (n_tx, T).
    alamouti_ofdm_map / unmap   Alamouti over OFDM: one subcarrier, two
                                consecutive OFDM blocks per symbol pair.
    SpaceTimeTransmitter        full chain: map -> code -> (remap) -> waveform.

chain_test.py
    Self-test of the coding schemes (shapes, energy, noiseless decoding).

Conventions shared with the Channel and Receiver modules
--------------------------------------------------------
* E|s|^2 = A for every constellation, so Es/N0 = A/N0.
* The OFDM IDFT is unitary: energy per time sample = energy per subcarrier.
* power_split=True (default) scales X by 1/sqrt(n_tx), so the total transmit
  energy per channel use is A for any number of antennas. The Receiver must be
  called with the same setting.
* Bits are mapped MSB first; Gray labels make BER ~ SER / log2(M) at high SNR.

Usage
-----
    from Transmitter.Transmitter import Constellation, Waveform, Transmitter
    from Transmitter.Coding_Schemes import Alamouti, SpaceTimeTransmitter

    const = Constellation("QAM", M=16, A=1.0)
    wf    = Waveform("OFDM", n_subcarriers=64, cyclic_prefix=16)
    tx    = SpaceTimeTransmitter(Transmitter(const, wf), Alamouti())
    X     = tx.run(bits, len(bits))        # (2, n_samples), ready for the channel

Known limitations
-----------------
* The cyclic prefix length is not checked against the channel memory
  (L_cp >= L - 1 is the user's responsibility).
* Square QAM only; non-square orders need a custom mapping.
* No pilot insertion or frame structure yet.

Run this file for a short demonstration:
    python Transmitter/welcome_code.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo():
    from Transmitter.Transmitter import Constellation, Waveform, Transmitter
    from Transmitter.Coding_Schemes import Alamouti, VBLAST, SpaceTimeTransmitter

    rng = np.random.default_rng(0)
    const = Constellation("QAM", M=16, A=1.0)
    print(f"16-QAM: {const.M} points, {const.bits_per_symbol} bits/symbol, "
          f"average energy {np.mean(np.abs(const.table) ** 2):.3f}")

    bits = rng.integers(0, 2, 1024)
    wf = Waveform("OFDM", n_subcarriers=64, cyclic_prefix=16)
    for scheme in (Alamouti(), VBLAST(2)):
        tx = SpaceTimeTransmitter(Transmitter(const, wf), scheme)
        X = tx.run(bits, len(bits))
        print(f"{type(scheme).__name__:9s} + OFDM(64, CP 16): {len(bits)} bits -> "
              f"X of shape {X.shape}")


if __name__ == "__main__":
    print("Transmitter module - EECE 640 Wireless Communication class project")
    demo()
