import numpy as np


class CodingScheme:
    """
    Base class for transmit-side coding schemes. Input is the output of
    Transmitter.run (a 1-D complex array of constellation symbols); output
    of encode() is a complex array of shape (n_tx, T): rows = antennas,
    columns = channel uses, i.e. the X in Y = HX + N.

    n_tx        : number of transmit antennas
    power_split : if True, each antenna is scaled by 1/sqrt(n_tx) so the
                  total energy per channel use stays A.
    """

    def __init__(self, n_tx, power_split=True):
        if not isinstance(n_tx, (int, np.integer)) or n_tx <= 0:
            raise ValueError("n_tx must be a positive integer.")
        self.n_tx = int(n_tx)
        self.power_split = power_split

    @staticmethod
    def _check_symbols(symbols):
        symbols = np.asarray(symbols)
        if symbols.ndim != 1 or symbols.size == 0:
            raise ValueError("symbols must be a nonempty 1-D array.")
        if not np.all(np.isfinite(symbols)):
            raise ValueError("symbols must contain finite values.")
        return symbols.astype(complex)

    def _scale(self, X):
        return X / np.sqrt(self.n_tx) if self.power_split else X

    def encode(self, symbols):
        raise NotImplementedError

    def channel_uses(self, n_symbols):
        """Number of channel uses (columns of X) needed for n_symbols symbols."""
        raise NotImplementedError


class Repetition(CodingScheme):
    """
    Repeats each symbol L times (x_l = x for l = 1..L) on a single antenna.
    Columns are ordered [s1 s1 ... s1, s2 s2 ... s2, ...]; the receiver can
    reshape(-1, L).
    """

    def __init__(self, L):
        if not isinstance(L, (int, np.integer)) or L <= 0:
            raise ValueError("L must be a positive integer.")
        super().__init__(n_tx=1)
        self.L = int(L)

    def encode(self, symbols):
        s = self._check_symbols(symbols)
        return self._scale(np.repeat(s, self.L)[None, :])

    def channel_uses(self, n_symbols):
        return n_symbols * self.L


class Alamouti(CodingScheme):
    """
    2-transmit-antenna Alamouti code. Symbols are taken in pairs (u1, u2):

        antenna 1:  u1   -u2*
        antenna 2:  u2    u1*      (columns = channel uses)

    Needs an even number of symbols.
    """

    def __init__(self, power_split=True):
        super().__init__(n_tx=2, power_split=power_split)

    def encode(self, symbols):
        s = self._check_symbols(symbols)
        self.channel_uses(len(s))

        s1, s2 = s[0::2], s[1::2]
        X = np.zeros((2, len(s)), dtype=complex)
        X[0, 0::2] = s1
        X[0, 1::2] = -np.conj(s2)
        X[1, 0::2] = s2
        X[1, 1::2] = np.conj(s1)

        return self._scale(X)

    def channel_uses(self, n_symbols):
        if n_symbols % 2 != 0:
            raise ValueError("Alamouti coding needs an even number of symbols.")
        return n_symbols


class VBLAST(CodingScheme):
    """
    V-BLAST spatial multiplexing: the symbol stream is split across n_tx
    antennas, so each channel use carries n_tx independent symbols.
    Needs a number of symbols that is a multiple of n_tx.
    """

    def __init__(self, n_tx, power_split=True):
        super().__init__(n_tx=n_tx, power_split=power_split)

    def encode(self, symbols):
        s = self._check_symbols(symbols)
        self.channel_uses(len(s))

        return self._scale(s.reshape(-1, self.n_tx).T.copy())

    def channel_uses(self, n_symbols):
        if n_symbols % self.n_tx != 0:
            raise ValueError(
                f"Number of symbols ({n_symbols}) must be a multiple of n_tx = {self.n_tx}."
            )
        return n_symbols // self.n_tx

# ======================================================================
# Non-coherent (differential) schemes under the same CodingScheme contract
# ======================================================================
# The reference encoders were written together with their detectors and live in
# Receiver/receiver.py (differential_encode, differential_alamouti_encode). These
# wrappers only expose them through the transmitter-side CodingScheme interface
# (symbols -> X of shape (n_tx, T)), so the controller can treat every scheme the
# same way. They are imported lazily so this file stays importable on its own.

class DifferentialPSK(CodingScheme):
    """
    Single-antenna differential PSK: d_0 = sqrt(A), d_n = d_{n-1} * s_n / sqrt(A)
    (receiver.differential_encode). Needs constant-modulus (PSK) symbols.
    One reference symbol is added: T = Ns + 1. Detected by Receiver.differential_detect.
    """

    def __init__(self):
        super().__init__(n_tx=1)

    def encode(self, symbols):
        from Receiver.receiver import differential_encode
        s = self._check_symbols(symbols)
        A = float(np.mean(np.abs(s) ** 2))
        return differential_encode(s, A)[None, :]

    def channel_uses(self, n_symbols):
        return n_symbols + 1


class DifferentialAlamouti(CodingScheme):
    """
    Differential (non-coherent) Alamouti code for 2 transmit antennas
    (receiver.differential_alamouti_encode). Needs an even number of PSK symbols;
    one 2-slot reference block is added: T = Ns + 2. With power_split=True the total
    energy per channel use is A (the encoder applies the split itself).
    Detected by Receiver.differential_alamouti_detect.
    """

    def __init__(self, power_split=True):
        super().__init__(n_tx=2, power_split=power_split)

    def encode(self, symbols):
        from Receiver.receiver import differential_alamouti_encode
        s = self._check_symbols(symbols)
        self.channel_uses(len(s))
        A = float(np.mean(np.abs(s) ** 2))
        return differential_alamouti_encode(s, A, self.power_split)

    def channel_uses(self, n_symbols):
        if n_symbols % 2 != 0:
            raise ValueError("Differential Alamouti needs an even number of symbols.")
        return n_symbols + 2


# ======================================================================
# Alamouti over OFDM: space-time code on each subcarrier across two OFDM blocks
# ======================================================================
# Alamouti needs the channel to be the same for the two slots of a pair. With OFDM
# over a multipath channel, adjacent SUBCARRIERS see different gains, but the SAME
# subcarrier in two consecutive OFDM blocks sees the same gain when the channel is
# block-static (MIMOLTapChannel / MIMORicianFlatChannel). The pair m therefore goes
# to subcarrier k = m mod n of blocks 2j and 2j+1, j = m // n, so that the receiver
# can run the ordinary Alamouti combiner on every subcarrier (exact, no approximation).

def alamouti_ofdm_map(X, n_subcarriers):
    """
    Reorder the columns of an Alamouti codeword matrix X (2, T) for OFDM: pair m
    (columns 2m, 2m+1) -> subcarrier m % n of OFDM blocks 2(m // n) and 2(m // n) + 1.
    Needs T to be a multiple of 2 * n_subcarriers (an even number of OFDM blocks).
    """
    X = np.asarray(X)
    n = int(n_subcarriers)
    if X.ndim != 2 or X.shape[1] % (2 * n) != 0:
        raise ValueError(
            "Alamouti over OFDM needs a multiple of 2 * n_subcarriers channel uses "
            "(an even number of OFDM blocks)."
        )
    m = np.arange(X.shape[1] // 2)
    j, k = m // n, m % n
    Y = np.empty_like(X)
    Y[:, 2 * j * n + k] = X[:, 2 * m]
    Y[:, (2 * j + 1) * n + k] = X[:, 2 * m + 1]
    return Y


def alamouti_ofdm_unmap(u, n_subcarriers):
    """Inverse of alamouti_ofdm_map for a 1-D (or (..., T)) array in block-major order."""
    u = np.asarray(u)
    n = int(n_subcarriers)
    m = np.arange(u.shape[-1] // 2)
    j, k = m // n, m % n
    out = np.empty_like(u)
    out[..., 2 * m] = u[..., 2 * j * n + k]
    out[..., 2 * m + 1] = u[..., (2 * j + 1) * n + k]
    return out


# ======================================================================
# Unified transmitter: bits -> symbols -> CodingScheme -> waveform per antenna
# ======================================================================

class SpaceTimeTransmitter:
    """
    Connects Transmitter (constellation + waveform) with a CodingScheme:

        bits --Transmitter.symbols--> s (Ns,) --scheme.encode--> X (n_tx, T)
             --[Alamouti + OFDM: alamouti_ofdm_map]--> --Waveform.modulate per row-->
             samples (n_tx, n_samples)

    This is the "code first, then OFDM per antenna" order: the space-time code acts on
    constellation symbols (one column = one channel use = one subcarrier for OFDM), and
    every antenna row gets its own IFFT and cyclic prefix. The output is the X of the
    channel model Y = H X + N and can be passed directly to AWGN_Channel,
    MIMORicianFlatChannel or MIMOLTapChannel. The receiver decodes it with the matching
    Receiver method (decode_mimo / decode_ofdm_mimo / decode_sc / decode_mlse_mimo for
    V-BLAST, decode_alamouti / decode_alamouti_ofdm, decode_repetition,
    decode_differential, decode_differential_alamouti) with the SAME power_split.

    transmitter : Transmitter.Transmitter (its constellation and waveform are used)
    scheme      : a CodingScheme instance
    alamouti_across_blocks : None (default) -> True exactly when the scheme is Alamouti
                  and the waveform is OFDM (see alamouti_ofdm_map).
    """

    def __init__(self, transmitter, scheme, alamouti_across_blocks=None):
        if not (hasattr(transmitter, "symbols") and hasattr(transmitter, "waveform")):
            raise TypeError("transmitter must be a Transmitter.Transmitter object.")
        if not isinstance(scheme, CodingScheme):
            raise TypeError("scheme must be a CodingScheme object.")
        self.transmitter = transmitter
        self.scheme = scheme
        self.waveform = transmitter.waveform
        self.constellation = transmitter.constellation
        self.n_tx = scheme.n_tx
        self.k = transmitter.k
        is_ofdm = self.waveform.name == "OFDM"
        if alamouti_across_blocks is None:
            alamouti_across_blocks = isinstance(scheme, Alamouti) and is_ofdm
        if alamouti_across_blocks and not (isinstance(scheme, Alamouti) and is_ofdm):
            raise ValueError("alamouti_across_blocks needs the Alamouti scheme and OFDM.")
        self.alamouti_across_blocks = bool(alamouti_across_blocks)

    def channel_uses(self, n_bits):
        """Columns of X (per antenna, before the waveform) for n_bits information bits."""
        if n_bits % self.k:
            raise ValueError(f"n_bits must be a multiple of log2(M) = {self.k}.")
        return self.scheme.channel_uses(n_bits // self.k)

    def encode(self, bits, N):
        """Symbol-domain codeword matrix X (n_tx, T), after the optional OFDM mapping."""
        X = self.scheme.encode(self.transmitter.symbols(bits, N))
        if self.alamouti_across_blocks:
            X = alamouti_ofdm_map(X, self.waveform.n_subcarriers)
        return X

    def run(self, bits, N):
        """Transmit samples (n_tx, n_samples): the waveform applied to every antenna row."""
        X = self.encode(bits, N)
        return np.vstack([self.waveform.modulate(row) for row in X])
