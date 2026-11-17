"""
EECE 640 - Wireless Communication, Class Project
MODULE: RECEIVER

Purpose
-------
Recovers the transmitted bits from the channel output. Every detector follows
the same three steps:

    1. waveform demodulation  (cyclic-prefix removal; DFT for OFDM)
    2. equalization / combining  -> symbol estimates z
    3. minimum-distance slicing and label lookup  -> bits

Coherent detectors assume perfect channel state information (the channel
coefficients returned by the Channel module); MMSE-type detectors also need N0.

File
----
receiver.py
    class Receiver(constellation, waveform)
        SISO, flat       zf_equalize, mmse_equalize (unbiased for slicing)
        SIMO, flat       mrc_combine, egc_combine, selection_combine
        Repetition       repetition_combine (N_r * L observations per symbol)
        Alamouti         alamouti_combine; alamouti_ofdm_combine over OFDM
        V-BLAST, flat    zf_mimo, mmse_mimo, sic_vblast (ordered SIC);
                         joint ML via mlse_detect_mimo with a 1-tap channel
        SC over ISI      design_equalizer / equalize_sc / detect_sc
                         (ZF or MMSE, linear or decision-feedback);
                         mlse_detect_mimo (Viterbi, M-algorithm, K-best,
                         channel-shortening prefilter)
        SC+CP / OFDM     ofdm_mimo_detect (per-subcarrier ZF / MMSE / SIC; SC-FDE)
        Non-coherent     differential_detect, differential_alamouti_detect,
                         energy_detect, fsk_detect
        Estimation       pilots, estimate_channel (LS, or LMMSE when N0 is given)
        Metrics          ber, ser
    Helpers: pilot_matrix, differential_encode, differential_alamouti_encode,
             fsk_encode, theory_ser_qam, monte_carlo

    Each detector has a decode_* wrapper that returns the bit stream in the
    transmitted order.

Conventions shared with the Transmitter and Channel modules
-----------------------------------------------------------
* Build the Receiver with the same Constellation and Waveform as the transmitter.
* Pass the same power_split setting as the coding scheme (default True).
* Y is (Nr, samples); H is (Nr, Nt) for flat channels or (Nr, Nt, L) for
  multipath, exactly as returned by the Channel module.
* MMSE estimates are un-biased before slicing so that QAM decisions are correct.

Usage
-----
    from Receiver.receiver import Receiver

    rx   = Receiver(const, wf)                       # same objects as the transmitter
    bits = rx.decode_mimo(Y, H, method="sic", N0=N0) # V-BLAST over a flat channel
    bits = rx.decode_ofdm_mimo(Y, h, method="mmse", N0=N0)   # MIMO-OFDM, multipath
    ber  = rx.ber(tx_bits, bits)

Known limitations
-----------------
* Channel estimation is implemented but not yet connected to a frame structure;
  all results assume perfect CSI.
* Hard decisions only (no log-likelihood ratios for a channel decoder).
* mmse_mimo returns the biased estimate by default; request unbiased=True when
  slicing QAM directly.
* Full Viterbi is limited to 2^16 trellis states.

Run this file for a short demonstration:
    python Receiver/welcome_code.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo():
    from Transmitter.Transmitter import Constellation
    from Receiver.receiver import Receiver, theory_ser_qam

    rng = np.random.default_rng(0)
    const = Constellation("QAM", M=16, A=1.0)
    rx = Receiver(const)

    EsN0_dB, n_bits = 14.0, 400_000
    N0 = const.A / 10 ** (EsN0_dB / 10)
    bits = rng.integers(0, 2, n_bits)
    s = const.table[rx.bits_to_indices(bits)]
    noise = np.sqrt(N0 / 2) * (rng.standard_normal(s.size) + 1j * rng.standard_normal(s.size))
    rbits = rx.decode(s + noise)

    print(f"16-QAM over AWGN at Es/N0 = {EsN0_dB:.0f} dB:")
    print(f"  simulated SER = {rx.ser(bits, rbits):.3e}, "
          f"theory SER = {theory_ser_qam(16, 10 ** (EsN0_dB / 10)):.3e}")
    print(f"  simulated BER = {rx.ber(bits, rbits):.3e}")


if __name__ == "__main__":
    print("Receiver module - EECE 640 Wireless Communication class project")
    demo()
