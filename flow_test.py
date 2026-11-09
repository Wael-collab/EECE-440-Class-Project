"""End-to-end BER/SER checks for OFDM and cyclic-prefix-free single carrier.

The flat-fading channels generate one independent gain per transmitted sample.
Single carrier equalizes after waveform demodulation; OFDM removes those
sample-wise gains before its FFT. Both use the channel gains returned by the
channel functions.
"""

import numpy as np

from Channel.AWGNChannel import AWGN_Channel
from Channel.LTapChannel import LTapChannel
from Channel.OFDMChannel import OFDM as LegacyOFDMChannel
from Channel.Rayleigh import Rayleigh_Channel
from Channel.RicianChannel import Rician_Channel
from Receiver.receiver import Receiver
from Transmitter.Transmitter import Constellation, Transmitter, Waveform

try:
    from Channel.AWSN import Alpha_Stable_Channel
except ModuleNotFoundError as exc:
    if exc.name != "aub_htp":
        raise
    Alpha_Stable_Channel = None


def report(label, bits, receiver, samples, gain=None, note=""):
    """Decode received samples and print their bit/symbol error rates."""
    decoded = receiver.decode(samples, gain)
    print(
        f"  {label:<19} BER={receiver.ber(bits, decoded):.6f}  "
        f"SER={receiver.ser(bits, decoded):.6f}  "
        f"bit errors={np.count_nonzero(bits != decoded)}{note}"
    )


rng = np.random.default_rng(7)
np.random.seed(7)  # Existing channel functions use NumPy's global RNG.
n_bits = 4096
bits = rng.integers(0, 2, n_bits)
constellation = Constellation("QAM", M=16, A=4)
ebn0_db = 10
n0 = (constellation.A / constellation.bits_per_symbol) / (10 ** (ebn0_db / 10))
l_taps = 4

print(f"16-QAM, {n_bits} bits, nominal Eb/N0={ebn0_db} dB, N0={n0:.4f}")

for name, waveform in (
    ("OFDM + CP", Waveform("OFDM", n_subcarriers=64, cyclic_prefix=16)),
    ("SC without CP", Waveform("SC")),
):
    transmitter = Transmitter(constellation, waveform)
    receiver = Receiver(constellation, waveform)
    transmitted = transmitter.run(bits, n_bits)

    print(f"\n{name}: {len(transmitted)} transmitted samples")
    clean_bits = receiver.decode(transmitted)
    assert np.array_equal(bits, clean_bits), f"{name}: noiseless round trip failed"
    report("Noiseless", bits, receiver, transmitted)

    noisy = AWGN_Channel(transmitted, n0)
    report("AWGN", bits, receiver, noisy)
    if waveform.name == "OFDM":
        ofdm_awgn_symbols = waveform.demodulate(noisy)

    faded, h = Rayleigh_Channel(transmitted, n0)
    if waveform.name == "SC":
        report("Rayleigh", bits, receiver, faded, h, " (coherent)")
    else:
        report("Rayleigh", bits, receiver, faded / h,
               note=" (sample-domain coherent)")

    faded, h = Rician_Channel(transmitted, n0, K=1.0)
    if waveform.name == "SC":
        report("Rician", bits, receiver, faded, h, " (coherent)")
    else:
        report("Rician", bits, receiver, faded / h,
               note=" (sample-domain coherent)")

    convolved, taps = LTapChannel(l_taps, transmitted, n0)
    convolved = convolved[:len(transmitted)]  # Drop the convolution tail.
    if waveform.name == "OFDM":
        # With CP >= L - 1, convolution is circular within each OFDM data block.
        gains = np.tile(
            np.fft.fft(taps, waveform.n_subcarriers),
            len(transmitted) // (waveform.n_subcarriers + waveform.cyclic_prefix),
        )
        report("L-tap", bits, receiver, convolved, gains, " (coherent)")
    else:
        report("L-tap", bits, receiver, convolved, note=" (unequalized ISI)")

    if Alpha_Stable_Channel is None:
        print("  Alpha-stable        skipped (aub_htp is not installed)")
    else:
        # gamma is a noise scale, not the AWGN variance N0.
        stable = Alpha_Stable_Channel(transmitted, alpha=1.8, gamma=0.1)
        report("Alpha-stable", bits, receiver, stable, note=" (gamma=0.1)")

# This legacy function performs its own OFDM modulation, CP, channel and FFT;
# passing an already-modulated waveform to it would apply OFDM twice.
legacy_bits = bits[:64 * constellation.bits_per_symbol]
legacy_symbols = Transmitter(constellation, Waveform("SC")).run(
    legacy_bits, len(legacy_bits)
)
legacy_received, legacy_taps = LegacyOFDMChannel(legacy_symbols, l_taps, n0)
legacy_gains = np.fft.fft(legacy_taps, len(legacy_symbols))
print("\nLegacy OFDMChannel (standalone 64-symbol block):")
report("OFDMChannel", legacy_bits, Receiver(constellation), legacy_received,
       legacy_gains, " (coherent)")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\nPlot skipped: matplotlib is not installed.")
else:
    plt.figure(figsize=(7, 6))
    plt.scatter(constellation.table.real, constellation.table.imag,
                marker="x", s=90, label="Ideal QAM points")
    plt.scatter(ofdm_awgn_symbols.real, ofdm_awgn_symbols.imag,
                s=12, alpha=0.35, label="OFDM + AWGN symbols")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(f"16-QAM received constellation, Eb/N0 = {ebn0_db} dB")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
