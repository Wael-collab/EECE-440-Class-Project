import numpy as np

from Transmitter.Transmitter import Constellation, Transmitter, Waveform
from Receiver.receiver import Receiver
from Channel.AWGNChannel import AWGN_Channel

# 1. Generate random input bits
rng = np.random.default_rng(7)
N = 4096
bits = rng.integers(0, 2, N)

# 2. Configure the transmitter and receiver
constellation = Constellation("QAM", M=16, A=4)
waveform = Waveform("OFDM", n_subcarriers=64, cyclic_prefix=16)
tx = Transmitter(constellation, waveform)
rx = Receiver(constellation, waveform)

# 3. Transmit: bits -> QAM symbols
waveform_tx = tx.run(bits, N)

print("Number of input bits:", len(bits))
print("Number of transmitted waveform samples:", len(waveform_tx))

# 4. TEST 1: No noise
bits_rx_clean = rx.decode(waveform_tx)
ber_clean = rx.ber(bits, bits_rx_clean)

print("\n--- ------------------------------------------------------------------ ---")


print("\n--- Noiseless test ---")
print("BER:", ber_clean)
print("All bits recovered correctly:", np.array_equal(bits, bits_rx_clean))
assert np.array_equal(bits, bits_rx_clean), "Noiseless waveform round trip failed."

# 5. TEST 2: Add AWGN noise
EbN0_dB = 10
k = constellation.bits_per_symbol
Es = constellation.A

N0 = (Es / k) / (10 ** (EbN0_dB / 10))

waveform_rx = AWGN_Channel(waveform_tx, N0)

# 6. Receive: noisy symbols -> recovered bits
bits_rx = rx.decode(waveform_rx)

ber = rx.ber(bits, bits_rx)
ser = rx.ser(bits, bits_rx)

print("\n--- ------------------------------------------------------------------ ---")

print("\n--- Noisy-channel test ---")
print("Eb/N0:", EbN0_dB, "dB")
print("BER:", ber)
print("SER:", ser)
print("Number of bit errors:", np.sum(bits != bits_rx))

# 7. Plot the received constellation
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Plot skipped: matplotlib is not installed.")
else:
    symbols_rx = rx.waveform.demodulate(waveform_rx)
    plt.figure(figsize=(7, 6))

    plt.scatter(
        constellation.table.real,
        constellation.table.imag,
        marker="x",
        s=90,
        label="Ideal QAM points"
    )

    plt.scatter(
        symbols_rx.real,
        symbols_rx.imag,
        s=12,
        alpha=0.35,
        label="Received symbols"
    )

    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(f"16-QAM received constellation, Eb/N0 = {EbN0_dB} dB")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
