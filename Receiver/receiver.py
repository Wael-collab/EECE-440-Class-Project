import math
import numpy as np

from Transmitter.Transmitter import Waveform


class Receiver:
    """
    Takes received waveform samples and turns them back into bits.
    Uses the same Constellation and Waveform configuration as the Transmitter;
    waveform demodulation removes any cyclic prefix before symbol detection.

    Besides the SISO detector it has the MIMO receivers matching Coding_Schemes
    (Y = H X + N, X of shape (n_tx, T)): scalar/MIMO ZF and MMSE, ordered SIC
    (V-BLAST), Alamouti combining and pilot-based channel estimation, plus
    non-coherent detection that needs neither the channel nor pilots:
    differential (DPSK, differential Alamouti), energy detection (OOK /
    amplitude levels) and orthogonal M-FSK.

    constellation.table[i] is the symbol for the label
    format(i, "0kb"), so the table index IS the label as an integer.
    """

    def __init__(self, constellation, waveform=None):
        self.constellation = constellation
        if waveform is None:
            waveform = Waveform("SC")
        if not isinstance(waveform, Waveform):
            raise TypeError("waveform must be a Waveform object.")
        self.waveform = waveform
        self.k = constellation.bits_per_symbol
        self.M = constellation.M
        self.table = np.asarray(constellation.table, dtype=complex)

        self.bit_table = np.array(
            [[(i >> (self.k - 1 - j)) & 1 for j in range(self.k)] for i in range(self.M)],
            dtype=int,
        )

    # ---------- ML detector ----------
    def _slice(self, s):
        """Minimum-distance slicer: closest constellation index for every entry of s (any shape)."""
        s = np.asarray(s, dtype=complex)
        d2 = np.abs(s[..., None] - self.table) ** 2
        return np.argmin(d2, axis=-1)

    def detect(self, y, h=None, N0=None):
        """
        Demodulate the waveform, then for each recovered symbol pick the
        closest constellation point.
        """
        y = self.waveform.demodulate(y)
        if h is not None:
            if N0 is None:
                y = y / np.asarray(h, dtype=complex)
            else:
                y = self.mmse_equalize(y, h, N0, unbiased=True)
        return self._slice(y)

    # ---------- Lookup-table decoder ----------
    def decode(self, y, h=None, N0=None):
        """Detect the symbols, then look up their bits. Returns a 1-D bit array."""
        idx = self.detect(y, h, N0)
        return self.bit_table[idx].ravel()

    # ---------- scalar (SISO) equalization ----------
    def zf_equalize(self, y, h):
        """Scalar zero forcing: x_hat = y / h."""
        return np.asarray(y, dtype=complex) / np.asarray(h, dtype=complex)

    def mmse_equalize(self, y, h, N0, unbiased=False):
        """
        Scalar MMSE equalizer for y = h*x + n, with E|x|^2 = Es = constellation.A.
        """
        if N0 < 0:
            raise ValueError("N0 must be non-negative.")
        y = np.asarray(y, dtype=complex)
        h = np.asarray(h, dtype=complex)
        g = np.abs(h) ** 2
        x_hat = np.conj(h) / (g + N0 / self.constellation.A) * y
        if unbiased:
            x_hat = x_hat * (g + N0 / self.constellation.A) / g
        return x_hat

    # ---------- SIMO diversity combining ----------

    def _simo_inputs(self, Y, H):
        """
        Prepare received signals and channel gains for SIMO combining.

        Y: (Nr, T) received signals.
        H: (Nr,) constant channel gains or (Nr, T) time-varying gains.

        Each received branch is passed through the waveform demodulator.
        """
        Y = np.asarray(Y, dtype=complex)
        H = np.asarray(H, dtype=complex)

        if Y.ndim == 1:
            Y = Y[None, :]

        if H.ndim == 1:
            H = H[:, None]

        if Y.ndim != 2 or H.ndim != 2:
            raise ValueError("Y and H must be 1-D or 2-D arrays.")

        if Y.shape[0] != H.shape[0]:
            raise ValueError("Y and H must have the same number of receive antennas.")

        if H.shape[1] not in (1, Y.shape[1]):
            raise ValueError("H must have either one column or T columns.")

        Y = np.vstack([self.waveform.demodulate(row) for row in Y])

        return Y, H

    # ---------- Maximum Ratio Combining (MRC) ----------

    def mrc_combine(self, Y, H):
        """
        Maximum Ratio Combining for SIMO.

        System model:
            y_r = h_r*x + n_r

        MRC formula:
            x_hat = sum(conj(h_r)*y_r) / sum(|h_r|^2)

        Y: (Nr, T) received signals.
        H: (Nr,) or (Nr, T) channel gains.

        Returns:
            Combined symbol estimates of shape (T,).
        """
        Y, H = self._simo_inputs(Y, H)

        numerator = np.sum(np.conj(H) * Y, axis=0)
        denominator = np.sum(np.abs(H) ** 2, axis=0)

        return numerator / np.maximum(denominator, 1e-300)

    def decode_mrc(self, Y, H):
        """MRC combining followed by symbol detection and bit decoding."""
        combined = self.mrc_combine(Y, H)
        return self.bit_table[self._slice(combined)].ravel()

    # ---------- Equal Gain Combining (EGC) ----------

    def egc_combine(self, Y, H):
        """
        Equal Gain Combining for SIMO.

        Each branch is phase-aligned using its channel estimate, then
        the aligned signals are combined with equal weights.

        Formula:
            x_hat = sum((conj(h_r)/|h_r|)*y_r) / sum(|h_r|)

        Y: (Nr, T) received signals.
        H: (Nr,) or (Nr, T) channel gains.

        Returns:
            Combined symbol estimates of shape (T,).
        """
        Y, H = self._simo_inputs(Y, H)

        magnitudes = np.abs(H)
        phase_weights = np.conj(H) / np.maximum(magnitudes, 1e-300)

        numerator = np.sum(phase_weights * Y, axis=0)
        denominator = np.sum(magnitudes, axis=0)

        return numerator / np.maximum(denominator, 1e-300)

    def decode_egc(self, Y, H):
        """EGC combining followed by symbol detection and bit decoding."""
        combined = self.egc_combine(Y, H)
        return self.bit_table[self._slice(combined)].ravel()

    # ---------- Selection Combining (SC) ----------

    def selection_combine(self, Y, H):
        """
        Selection Combining for SIMO.

        Selects the receive antenna with the highest instantaneous
        channel gain for each symbol.

        Selection rule:
            r_best = argmax(|h_r|^2)

        Equalization:
            x_hat = y_best / h_best

        Y: (Nr, T) received signals.
        H: (Nr,) or (Nr, T) channel gains.

        Returns:
            Combined symbol estimates of shape (T,).
        """
        Y, H = self._simo_inputs(Y, H)

        best_branch = np.argmax(np.abs(H) ** 2, axis=0)

        columns = np.arange(Y.shape[1])

        selected_y = Y[best_branch, columns]
        selected_h = H[best_branch, columns]

        return selected_y / np.where(
            np.abs(selected_h) > 0,
            selected_h,
            1.0
        )

    def decode_selection(self, Y, H):
        """Selection combining followed by symbol detection and bit decoding."""
        combined = self.selection_combine(Y, H)
        return self.bit_table[self._slice(combined)].ravel()

    # ---------- MIMO helpers ----------
    def _mimo_inputs(self, Y, H, power_split):
        """
        Check shapes and return (Y, H_eff) for a flat MIMO channel Y = H X + N.
        """
        Y = np.atleast_2d(np.asarray(Y, dtype=complex))
        H = np.asarray(H, dtype=complex)
        if H.ndim == 3:
            if H.shape[2] != 1:
                raise ValueError(
                    "Only flat MIMO channels are supported (L = 1); a multi-tap "
                    "channel has ISI and needs an equalizer first."
                )
            H = H[:, :, 0]
        if H.ndim != 2:
            raise ValueError("H must have shape (Nr, Nt).")
        if Y.ndim != 2 or Y.shape[0] != H.shape[0]:
            raise ValueError("Y must have shape (Nr, T) with Nr = H.shape[0].")
        Y = np.vstack([self.waveform.demodulate(row) for row in Y])
        H_eff = H / np.sqrt(H.shape[1]) if power_split else H
        return Y, H_eff

    @staticmethod
    def _stream(idx):
        """(Nt, T) layer indices -> 1-D stream in the order VBLAST.encode consumed them."""
        return np.asarray(idx).T.reshape(-1)

    # ---------- MIMO zero forcing ----------
    def zf_mimo(self, Y, H, power_split=True):
        """
        MIMO zero-forcing detection.
        """
        Y, H = self._mimo_inputs(Y, H, power_split)
        if H.shape[0] < H.shape[1]:
            raise ValueError("Zero forcing needs Nr >= Nt.")
        return np.linalg.pinv(H) @ Y

    # ---------- MIMO MMSE ----------
    def mmse_mimo(self, Y, H, N0, power_split=True, unbiased=False):
        """
        MIMO MMSE detection.
        """
        if N0 < 0:
            raise ValueError("N0 must be non-negative.")
        Y, H = self._mimo_inputs(Y, H, power_split)
        Nt = H.shape[1]
        Hh = H.conj().T
        W = np.linalg.solve(Hh @ H + (N0 / self.constellation.A) * np.eye(Nt), Hh)
        S = W @ Y
        if unbiased:
            S = S / np.diag(W @ H)[:, None]
        return S

    # ---------- SIC / V-BLAST ----------
    def sic_vblast(self, Y, H, N0=None, power_split=True, ordered=True):
        """
        Ordered successive interference cancellation (V-BLAST detection).
        """
        Y, H = self._mimo_inputs(Y, H, power_split)
        Nr, Nt = H.shape
        if N0 is None and Nr < Nt:
            raise ValueError("Zero-forcing SIC needs Nr >= Nt; pass N0 for MMSE nulling.")
        reg = 0.0 if N0 is None else N0 / self.constellation.A

        Yk = Y.copy()
        remaining = list(range(Nt))
        idx = np.zeros((Nt, Y.shape[1]), dtype=int)
        order = []

        while remaining:
            Hk = H[:, remaining]
            Hh = Hk.conj().T
            G = np.linalg.inv(Hh @ Hk + reg * np.eye(len(remaining)))
            W = G @ Hh
            pick = int(np.argmin(np.real(np.diag(G)))) if ordered else 0
            layer = remaining[pick]

            s_hat = (W[pick] @ Yk) / (W[pick] @ Hk[:, pick])
            idx[layer] = self._slice(s_hat)

            Yk = Yk - np.outer(H[:, layer], self.table[idx[layer]])
            remaining.pop(pick)
            order.append(layer)

        return idx, order

    # ---------- Alamouti combining ----------
    def alamouti_combine(self, Y, H, N0=None, power_split=True):
        """
        Alamouti combining for the 2-antenna code in Coding_Schemes.Alamouti.
        """
        Y, H = self._mimo_inputs(Y, H, power_split)
        if H.shape[1] != 2:
            raise ValueError("Alamouti combining needs H with Nt = 2 columns.")
        if Y.shape[1] % 2 != 0:
            raise ValueError("Alamouti needs an even number of channel uses.")

        y1, y2 = Y[:, 0::2], Y[:, 1::2]
        h1, h2 = H[:, [0]], H[:, [1]]
        z1 = np.sum(np.conj(h1) * y1 + h2 * np.conj(y2), axis=0)
        z2 = np.sum(np.conj(h2) * y1 - h1 * np.conj(y2), axis=0)
        g = np.sum(np.abs(H) ** 2)
        if N0 is not None:
            if N0 < 0:
                raise ValueError("N0 must be non-negative.")
            g = g + N0 / self.constellation.A

        u = np.empty(Y.shape[1], dtype=complex)
        u[0::2], u[1::2] = z1 / g, z2 / g
        return u

    # ---------- bit-level wrappers for the coding schemes ----------
    def decode_mimo(self, Y, H, method="mmse", N0=None, power_split=True):
        """
        Detect a V-BLAST transmission and return the bit stream.
        """
        method = method.lower()
        if method == "zf":
            idx = self._slice(self.zf_mimo(Y, H, power_split))
        elif method == "mmse":
            if N0 is None:
                raise ValueError("MMSE detection needs N0.")
            idx = self._slice(self.mmse_mimo(Y, H, N0, power_split, unbiased=True))
        elif method == "sic":
            idx, _ = self.sic_vblast(Y, H, N0, power_split)
        else:
            raise ValueError("method must be 'zf', 'mmse' or 'sic'.")
        return self.bit_table[self._stream(idx)].ravel()

    def decode_alamouti(self, Y, H, N0=None, power_split=True):
        """Alamouti-combine, slice and return the bit stream."""
        u = self.alamouti_combine(Y, H, N0, power_split)
        return self.bit_table[self._slice(u)].ravel()

    # ---------- pilot-based channel estimation ----------
    def pilots(self, n_tx, n_pilots=None, power_split=True):
        """Pilot matrix with the same energy rules as the transmitter."""
        return pilot_matrix(n_tx, n_pilots, self.constellation.A, power_split)

    def estimate_channel(self, Yp, Xp, N0=None, prior_var=1.0):
        """
        Estimate the flat MIMO channel from known pilots, Yp = H Xp + N.
        """
        Yp = np.atleast_2d(np.asarray(Yp, dtype=complex))
        Xp = np.atleast_2d(np.asarray(Xp, dtype=complex))
        Nt, Tp = Xp.shape
        if Yp.shape[1] != Tp:
            raise ValueError("Yp and Xp must have the same number of pilot columns.")
        if Tp < Nt:
            raise ValueError("Need at least as many pilot columns as transmit antennas.")

        A = Xp @ Xp.conj().T
        if N0 is not None:
            if N0 < 0 or prior_var <= 0:
                raise ValueError("N0 must be >= 0 and prior_var > 0.")
            A = A + (N0 / prior_var) * np.eye(Nt)
        B = Yp @ Xp.conj().T
        return np.linalg.solve(A.T, B.T).T

    # ---------- non-coherent detection ----------
    def _require_psk(self):
        """Differential schemes require a constant-modulus constellation."""
        mags = np.abs(self.table)
        if not np.allclose(mags, mags[0]):
            raise ValueError(
                "Differential (non-coherent) detection needs a constant-modulus "
                "constellation such as PSK; this one has points of different magnitude."
            )

    def _slice_phase(self, z):
        """Closest constellation point by phase."""
        z = np.asarray(z, dtype=complex)
        return np.argmax(np.real(np.conj(self.table) * z[..., None]), axis=-1)

    def differential_detect(self, y):
        """
        Differential (DPSK) detection.
        """
        self._require_psk()
        y = np.atleast_2d(np.asarray(y, dtype=complex))
        y = np.vstack([self.waveform.demodulate(row) for row in y])
        if y.shape[1] < 2:
            raise ValueError("Need the reference sample plus at least one data sample.")
        z = np.sum(y[:, 1:] * np.conj(y[:, :-1]), axis=0)
        return self._slice_phase(z)

    def decode_differential(self, y):
        """Differential detection followed by the bit lookup."""
        return self.bit_table[self.differential_detect(y)].ravel()

    def differential_alamouti_detect(self, Y):
        """
        Non-coherent differential Alamouti detection.
        """
        self._require_psk()
        Y = np.atleast_2d(np.asarray(Y, dtype=complex))
        Y = np.vstack([self.waveform.demodulate(row) for row in Y])
        if Y.shape[1] % 2 != 0 or Y.shape[1] < 4:
            raise ValueError("Y needs an even number of columns: reference block plus at least one data block.")

        c1, c2 = Y[:, 0::2], Y[:, 1::2]
        p1, p2, q1, q2 = c1[:, :-1], c2[:, :-1], c1[:, 1:], c2[:, 1:]
        E = np.maximum(np.sum(np.abs(p1) ** 2 + np.abs(p2) ** 2, axis=0), 1e-300)
        s1 = np.sum(np.conj(p1) * q1 + p2 * np.conj(q2), axis=0) / E
        s2 = np.sum(np.conj(p2) * q1 - p1 * np.conj(q2), axis=0) / E

        scale = np.sqrt(2 * self.constellation.A)
        idx = np.empty(2 * s1.size, dtype=int)
        idx[0::2] = self._slice_phase(scale * s1)
        idx[1::2] = self._slice_phase(scale * s2)
        return idx

    def decode_differential_alamouti(self, Y):
        """Differential Alamouti detection followed by bit lookup."""
        return self.bit_table[self.differential_alamouti_detect(Y)].ravel()

    # ---------- non-coherent: energy detection ----------
    def energy_detect(self, y, N0=None, fading_var=1.0):
        """
        Energy (envelope) detector for OOK and amplitude constellations.
        """
        levels = np.abs(self.table) ** 2
        if len(np.unique(np.round(levels, 12))) != self.M:
            raise ValueError(
                "Energy detection needs every constellation point to have a different "
                "magnitude (e.g. OOK or amplitude levels); this constellation does not."
            )
        y = np.atleast_2d(np.asarray(y, dtype=complex))
        y = np.vstack([self.waveform.demodulate(row) for row in y])
        Nr = y.shape[0]
        E = np.sum(np.abs(y) ** 2, axis=0)

        if fading_var is None:
            n0 = 0.0 if N0 is None else N0
            return np.argmin(np.abs(E[:, None] / Nr - (levels[None, :] + n0)), axis=1)

        if N0 is None or N0 <= 0:
            raise ValueError("Fading ML energy detection needs N0 > 0.")
        if fading_var <= 0:
            raise ValueError("fading_var must be positive (or None for plain AWGN).")
        m = N0 + fading_var * levels
        return np.argmax(-Nr * np.log(m)[None, :] - E[:, None] / m[None, :], axis=1)

    def decode_energy(self, y, N0=None, fading_var=1.0):
        """Energy detection followed by bit lookup."""
        return self.bit_table[self.energy_detect(y, N0, fading_var)].ravel()

    # ---------- non-coherent: orthogonal M-FSK ----------
    def fsk_detect(self, Y):
        """
        Non-coherent detection of orthogonal M-FSK.
        """
        Y = np.asarray(Y, dtype=complex)
        if Y.ndim == 2:
            Y = Y[None]
        if Y.ndim != 3 or Y.shape[1] != self.M:
            raise ValueError("Y must have shape (M, T) or (Nr, M, T) with M = constellation.M tones.")
        return np.argmax(np.sum(np.abs(Y) ** 2, axis=0), axis=0)

    def decode_fsk(self, Y):
        """Non-coherent FSK detection followed by bit lookup."""
        return self.bit_table[self.fsk_detect(Y)].ravel()

    # ---------- helpers ----------
    def bits_to_indices(self, bits):
        """Group bits into chunks of k and turn each chunk into an integer."""
        bits = np.asarray(bits, dtype=int).ravel()
        if len(bits) % self.k != 0:
            raise ValueError("Number of bits must be a multiple of log2(M).")
        w = 1 << np.arange(self.k - 1, -1, -1)
        return bits.reshape(-1, self.k) @ w

    def ber(self, tx_bits, rx_bits):
        """Fraction of bits that came out wrong."""
        tx_bits, rx_bits = np.asarray(tx_bits), np.asarray(rx_bits)
        if tx_bits.shape != rx_bits.shape:
            raise ValueError("Bit arrays must have the same length.")
        return np.mean(tx_bits != rx_bits)

    def ser(self, tx_bits, rx_bits):
        """Fraction of symbols that came out wrong."""
        return np.mean(self.bits_to_indices(tx_bits) != self.bits_to_indices(rx_bits))


def pilot_matrix(n_tx, n_pilots=None, Es=1.0, power_split=True):
    """
    Orthogonal pilot matrix Xp of shape (n_tx, n_pilots).
    """
    if not isinstance(n_tx, (int, np.integer)) or n_tx <= 0:
        raise ValueError("n_tx must be a positive integer.")
    n_pilots = n_tx if n_pilots is None else n_pilots
    if not isinstance(n_pilots, (int, np.integer)) or n_pilots < n_tx:
        raise ValueError("n_pilots must be an integer >= n_tx.")
    if Es <= 0:
        raise ValueError("Es must be positive.")
    t, k = np.arange(n_tx)[:, None], np.arange(n_pilots)[None, :]
    P = np.exp(-2j * np.pi * t * k / n_pilots)
    return P * np.sqrt(Es / n_tx) if power_split else P * np.sqrt(Es)


def differential_encode(symbols, A=1.0):
    """
    Differential PSK encoder.
    """
    s = np.asarray(symbols, dtype=complex).ravel()
    if A <= 0:
        raise ValueError("A must be positive.")
    if not np.allclose(np.abs(s) ** 2, A):
        raise ValueError("Differential encoding needs constant-modulus symbols with |s|^2 = A (PSK).")
    d = np.empty(s.size + 1, dtype=complex)
    d[0] = np.sqrt(A)
    d[1:] = np.sqrt(A) * np.exp(1j * np.cumsum(np.angle(s)))
    return d


def differential_alamouti_encode(symbols, Es=1.0, power_split=True):
    """
    Differential Alamouti encoder.
    """
    u = np.asarray(symbols, dtype=complex).ravel()
    if Es <= 0:
        raise ValueError("Es must be positive.")
    if u.size == 0 or u.size % 2 != 0:
        raise ValueError("Need a non-empty even number of symbols.")
    if not np.allclose(np.abs(u) ** 2, Es):
        raise ValueError("Differential Alamouti needs constant-modulus PSK symbols.")

    a = u / np.sqrt(Es) / np.sqrt(2)
    P = a.size // 2
    r = 1 / np.sqrt(2)
    U = np.array([[r, -r], [r, r]], dtype=complex)
    scale = np.sqrt(Es) if power_split else np.sqrt(2 * Es)
    X = np.empty((2, 2 * (P + 1)), dtype=complex)
    X[:, 0:2] = scale * U
    for p in range(P):
        s1, s2 = a[2 * p], a[2 * p + 1]
        U = U @ np.array([[s1, -np.conj(s2)], [s2, np.conj(s1)]])
        X[:, 2 * (p + 1):2 * (p + 1) + 2] = scale * U
    return X


def fsk_encode(indices, M, Es=1.0):
    """
    Orthogonal M-FSK encoder in the correlator domain.
    """
    idx = np.asarray(indices, dtype=int).ravel()
    if M < 2 or (M & (M - 1)) != 0:
        raise ValueError("M must be a power of 2.")
    if Es <= 0:
        raise ValueError("Es must be positive.")
    if idx.size == 0 or idx.min() < 0 or idx.max() >= M:
        raise ValueError("indices must be a non-empty array of integers in 0..M-1.")
    X = np.zeros((M, idx.size), dtype=complex)
    X[idx, np.arange(idx.size)] = np.sqrt(Es)
    return X


def qfunc(x):
    """Q-function: tail probability of a standard Gaussian."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def theory_ser_qam(M, EsN0):
    """
    Exact SER for square M-QAM over AWGN.
    """
    p = 2 * (1 - 1 / math.sqrt(M)) * qfunc(math.sqrt(3 * EsN0 / (M - 1)))
    return 1 - (1 - p) ** 2


def monte_carlo(constellation, channel, EbN0_dB, n_bits=100_000, n_trials=20, seed=None):
    """
    Monte Carlo BER/SER sweep.
    """
    rng = np.random.default_rng(seed)
    rx = Receiver(constellation)
    k, Es = constellation.bits_per_symbol, constellation.A

    n_bits -= n_bits % k

    out = {"EbN0_dB": [], "ber_mean": [], "ber_var": [], "ser_mean": [], "ser_var": []}

    for ebn0_db in EbN0_dB:
        N0 = (Es / k) / (10 ** (ebn0_db / 10))

        bers, sers = [], []
        for _ in range(n_trials):
            bits = rng.integers(0, 2, n_bits)
            idx = rx.bits_to_indices(bits)

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

