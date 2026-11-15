
"""
EECE 640 - Wireless Communication Class Project
MODULE: CHANNEL

This module simulates wireless signal propagation and noise.

Implemented Channel Models:
1. AWGNChannel.py          - Additive White Gaussian Noise
2. Rayleigh.py             - SISO flat Rayleigh fading
3. RicianChannel.py        - SISO flat Rician fading
4. MIMORicianFlatChannel.py - MIMO flat Rician fading (you can get MIMO Rayleigh by setting K=0) 
5. OFDMChannel.py          - Multipath channel and OFDM
6. MIMOLTapChannel.py      - MIMO multipath Rayleigh fading
7. AWSN.py                 - Alpha-stable impulsive noise

General Input:
    Transmitted symbols and channel parameters.

General Output:
    Received symbols and, when applicable, channel coefficients.

The generated channel coefficients can be passed to the
Receiver module for equalization and signal detection.
"""

print("Channel Module - EECE 640")
print("Wireless channel simulation models are available.")
