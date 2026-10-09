import math
import numpy as np


class Receiver:
    """
    Takes the noisy received symbols and turns them back into bits.
    Uses the same Constellation object the Transmitter used.

    constellation.table[i] is the symbol for the label
    format(i, "0kb"), so the table index IS the label as an integer.
    """

    def __init__(self, constellation):
        self.constellation = constellation
        self.k = constellation.bits_per_symbol   # bits per symbol
        self.M = constellation.M                 # number of constellation points
        self.table = np.asarray(constellation.table, dtype=complex)

        # Lookup table: symbol index -> its k bits (MSB first). Shape (M, k).
        # e.g. for 4 points: 0 -> [0,0], 1 -> [0,1], 2 -> [1,0], 3 -> [1,1]
        self.bit_table = np.array(
            [[(i >> (self.k - 1 - j)) & 1 for j in range(self.k)] for i in range(self.M)],
            dtype=int,
        )

    # ---------- ML detector ----------
    def detect(self, y, h=None):
        """
        Minimum-distance detection: for each received sample, pick the
        closest constellation point. Returns symbol INDICES (0..M-1), not bits.

        h (optional): channel gain if we know it (coherent flat fading).
        If y = h*x + n, we just divide by h to undo the channel, then do
        the usual closest-point search. Leave h=None for plain AWGN.
        """
        y = np.asarray(y, dtype=complex).ravel()
        if h is not None:
            y = y / np.asarray(h, dtype=complex)   # undo the channel

        # distance from every sample to every constellation point
        # shape: (n_symbols, M)
        d2 = np.abs(y[:, None] - self.table[None, :]) ** 2

        # closest point wins
        return np.argmin(d2, axis=1)

    # ---------- Lookup-table decoder ----------
    def decode(self, y, h=None):
        """Detect the symbols, then look up their bits. Returns a 1-D bit array."""
        idx = self.detect(y, h)
        return self.bit_table[idx].ravel()

    # ---------- helpers ----------
    def bits_to_indices(self, bits):
        """Group bits into chunks of k and turn each chunk into an integer (the symbol index)."""
        bits = np.asarray(bits, dtype=int).ravel()
        if len(bits) % self.k != 0:
            raise ValueError("Number of bits must be a multiple of log2(M).")
        w = 1 << np.arange(self.k - 1, -1, -1)   # binary weights, e.g. [4,2,1] for k=3
        return bits.reshape(-1, self.k) @ w

    def ber(self, tx_bits, rx_bits):
        """Fraction of bits that came out wrong."""
        tx_bits, rx_bits = np.asarray(tx_bits), np.asarray(rx_bits)
        if tx_bits.shape != rx_bits.shape:
            raise ValueError("Bit arrays must have the same length.")
        return np.mean(tx_bits != rx_bits)

    def ser(self, tx_bits, rx_bits):
        """Fraction of symbols that came out wrong (one wrong bit = whole symbol wrong)."""
        return np.mean(self.bits_to_indices(tx_bits) != self.bits_to_indices(rx_bits))


def qfunc(x):
    """Q-function: tail probability of a standard Gaussian."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def theory_ser_qam(M, EsN0):
    """
    Exact SER for square M-QAM (Gray mapping) over AWGN. EsN0 is linear, not dB.
    Use it to sanity-check the simulation.
    """
    p = 2 * (1 - 1 / math.sqrt(M)) * qfunc(math.sqrt(3 * EsN0 / (M - 1)))
    return 1 - (1 - p) ** 2


def monte_carlo(constellation, channel, EbN0_dB, n_bits=100_000, n_trials=20, seed=None):
    """
    Monte Carlo BER/SER sweep.

    channel  : function(Data, Nzero) -> noisy data (e.g. your AWGN_Channel)
    EbN0_dB  : list of Eb/N0 points in dB
    n_bits   : bits per trial
    n_trials : how many times we repeat each point (needed for the variance)

    Returns a dict with mean + variance (across trials) of BER and SER at each point.
    """
    rng = np.random.default_rng(seed)
    rx = Receiver(constellation)
    k, Es = constellation.bits_per_symbol, constellation.A

    n_bits -= n_bits % k   # need whole symbols, so drop any leftover bits

    out = {"EbN0_dB": [], "ber_mean": [], "ber_var": [], "ser_mean": [], "ser_var": []}

    for ebn0_db in EbN0_dB:
        # dB -> linear, then get the noise level: Eb = Es/k, N0 = Eb / (Eb/N0)
        N0 = (Es / k) / (10 ** (ebn0_db / 10))

        bers, sers = [], []
        for _ in range(n_trials):
            bits = rng.integers(0, 2, n_bits)
            idx = rx.bits_to_indices(bits)

            # same symbols the Transmitter would make, just done in one vectorized step
            y = channel(constellation.table[idx], N0)

            rbits = rx.decode(y)
            bers.append(rx.ber(bits, rbits))
            sers.append(rx.ser(bits, rbits))

        out["EbN0_dB"].append(ebn0_db)
        out["ber_mean"].append(np.mean(bers))
        out["ber_var"].append(np.var(bers, ddof=1))
        out["ser_mean"].append(np.mean(sers))
        out["ser_var"].append(np.var(sers, ddof=1))

    return {key: np.array(v) for key, v in out.items()}
