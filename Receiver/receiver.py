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
    def _slice(self, s):
        """Minimum-distance slicer: closest constellation index for every entry of s (any shape)."""
        s = np.asarray(s, dtype=complex)
        d2 = np.abs(s[..., None] - self.table) ** 2     # (..., M)
        return np.argmin(d2, axis=-1)

    def detect(self, y, h=None, N0=None):
        """
        Demodulate the waveform, then for each recovered symbol pick the
        closest constellation point. Returns symbol INDICES (0..M-1), not bits.

        h (optional): channel gain if we know it (coherent flat fading).
        If y = h*x + n, we just divide by h to undo the channel (zero forcing),
        then do the usual closest-point search. Leave h=None for plain AWGN.

        N0 (optional): noise variance. If both h and N0 are given, the scalar
        MMSE equalizer is used instead of zero forcing.
        """
        y = self.waveform.demodulate(y)
        if h is not None:
            if N0 is None:
                y = y / np.asarray(h, dtype=complex)    # zero forcing: undo the channel
            else:
                y = self.mmse_equalize(y, h, N0, unbiased=True)
        return self._slice(y)

    # ---------- Lookup-table decoder ----------
    def decode(self, y, h=None, N0=None):
        """Detect the symbols, then look up their bits. Returns a 1-D bit array.
        Pass N0 together with h to use the scalar MMSE equalizer instead of ZF."""
        idx = self.detect(y, h, N0)
        return self.bit_table[idx].ravel()

    # ---------- scalar (SISO) equalization ----------
    def zf_equalize(self, y, h):
        """Scalar zero forcing: x_hat = y / h."""
        return np.asarray(y, dtype=complex) / np.asarray(h, dtype=complex)

    def mmse_equalize(self, y, h, N0, unbiased=False):
        """
        Scalar MMSE equalizer for y = h*x + n, with E|x|^2 = Es = constellation.A
        and n ~ CN(0, N0) (same N0 as the channel functions):

            x_hat = conj(h) / (|h|^2 + N0/Es) * y

        y and h can be any matching shapes, per-symbol or per-sample gains
        (e.g. the h returned by Rayleigh_Channel / Rician_Channel), and y can be
        symbols or time-domain samples.

        The textbook MMSE output is shrunk towards 0 (biased). unbiased=True
        rescales by |h|^2 / (|h|^2 + N0/Es) so the output is centred on the
        constellation points, which is what the minimum-distance slicer needs
        for QAM. detect(y, h, N0) uses the unbiased version.
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

    # ---------- MIMO helpers ----------
    def _mimo_inputs(self, Y, H, power_split):
        """
        Check shapes and return (Y, H_eff) for a flat MIMO channel Y = H X + N.

        Y : (Nr, T) received matrix (a 1-D array is treated as Nr = 1). Each row
            goes through the waveform demodulator (a no-op for plain SC).
        H : (Nr, Nt) or (Nr, Nt, 1), e.g. the h returned by MIMOLTapChannel with
            L = 1. The channel must be constant over the block, as in
            MIMOLTapChannel, so it commutes with the CP removal and FFT.
        power_split : must match the CodingScheme. If the transmitter scaled X
            by 1/sqrt(n_tx), the channel the symbols actually see is H/sqrt(n_tx).
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
        MIMO zero-forcing detection: S_hat = pinv(H) Y = (H^H H)^-1 H^H Y.
        Needs Nr >= Nt. Returns the (Nt, T) symbol estimates (before slicing).
        """
        Y, H = self._mimo_inputs(Y, H, power_split)
        if H.shape[0] < H.shape[1]:
            raise ValueError("Zero forcing needs Nr >= Nt.")
        return np.linalg.pinv(H) @ Y

    # ---------- MIMO MMSE ----------
    def mmse_mimo(self, Y, H, N0, power_split=True, unbiased=False):
        """
        MIMO MMSE detection:  S_hat = (H^H H + (N0/Es) I)^-1 H^H Y,
        with Es = constellation.A and N0 the per-receive-antenna noise variance.
        Works for any Nr, including Nr < Nt. Returns the (Nt, T) estimates.

        unbiased=True divides each layer by the diagonal of W H (its gain on its
        own symbol) so the outputs are centred on the constellation points.
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

        Repeat Nt times: (1) build the nulling filters for the layers that are
        left, (2) pick the layer with the smallest error (best post-detection
        SNR) when ordered=True, else just the next one, (3) estimate it and
        slice it to the nearest constellation point, (4) subtract its
        contribution H[:, j] * s_j from Y and drop that column from H.

        N0=None  -> zero-forcing nulling (needs Nr >= Nt).
        N0 given -> MMSE nulling (unbiased before slicing).

        Because H is constant over the block, the detection order is the same
        for every channel use, so everything is vectorized over the T columns.

        Returns (idx, order): idx is (Nt, T) symbol indices in the original
        antenna order, order is the list of layers in the order they were detected.
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
            W = G @ Hh                                   # nulling filters, one row per layer
            pick = int(np.argmin(np.real(np.diag(G)))) if ordered else 0
            layer = remaining[pick]

            # estimate, remove the bias, slice
            s_hat = (W[pick] @ Yk) / (W[pick] @ Hk[:, pick])
            idx[layer] = self._slice(s_hat)

            # cancel this layer from the received vector
            Yk = Yk - np.outer(H[:, layer], self.table[idx[layer]])
            remaining.pop(pick)
            order.append(layer)

        return idx, order

    # ---------- Alamouti combining ----------
    def alamouti_combine(self, Y, H, N0=None, power_split=True):
        """
        Alamouti combining for the 2-antenna code in Coding_Schemes.Alamouti
        (antenna 1: u1, -u2*; antenna 2: u2, u1*), with Nr receive antennas.

            z1 = sum_r [ conj(h_r1) y_r1 + h_r2 conj(y_r2) ]
            z2 = sum_r [ conj(h_r2) y_r1 - h_r1 conj(y_r2) ]
            g  = sum_r (|h_r1|^2 + |h_r2|^2)

        y_r1, y_r2 are the two consecutive channel uses of receive antenna r.
        Then u_hat = z / g (zero forcing) or z / (g + N0/Es) (scalar MMSE) when
        N0 is given (unbiased MMSE is identical to z / g, so for slicing both
        give the same decisions; the MMSE form matters for the shrunk output).

        H is (Nr, 2), constant over the block. Returns the length-T symbol
        stream u1, u2, u3, u4, ... ready for slicing.
        """
        Y, H = self._mimo_inputs(Y, H, power_split)
        if H.shape[1] != 2:
            raise ValueError("Alamouti combining needs H with Nt = 2 columns.")
        if Y.shape[1] % 2 != 0:
            raise ValueError("Alamouti needs an even number of channel uses.")

        y1, y2 = Y[:, 0::2], Y[:, 1::2]
        h1, h2 = H[:, [0]], H[:, [1]]                    # (Nr, 1)
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
        method: "zf", "mmse" (needs N0) or "sic" (MMSE-SIC if N0 is given, else ZF-SIC).
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
        """Pilot matrix (n_tx, n_pilots) with the same energy rules as the data; see pilot_matrix()."""
        return pilot_matrix(n_tx, n_pilots, self.constellation.A, power_split)

    def estimate_channel(self, Yp, Xp, N0=None, prior_var=1.0):
        """
        Estimate the flat MIMO channel from known pilots, Yp = H Xp + N.

        Yp : (Nr, Tp) received pilots      Xp : (Nt, Tp) transmitted pilots
        (1-D inputs are treated as one antenna.) Needs Tp >= Nt and Xp of full
        row rank. Xp is the pilot matrix as actually transmitted, i.e. including
        any 1/sqrt(n_tx) power split; the returned H_hat is therefore the channel
        H itself, so pass it to the detectors with the same power_split as before.

        N0=None  -> least squares:  H_hat = Yp Xp^H (Xp Xp^H)^-1
        N0 given -> LMMSE for i.i.d. CN(0, prior_var) entries (Rayleigh):
                    H_hat = Yp Xp^H (Xp Xp^H + (N0/prior_var) I)^-1
        Returns H_hat of shape (Nr, Nt).
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
        return np.linalg.solve(A.T, B.T).T               # B A^-1

    # ---------- non-coherent detection (no channel knowledge, no pilots) ----------
    def _require_psk(self):
        """Differential schemes carry the data in the phase, so the constellation must be constant-modulus."""
        mags = np.abs(self.table)
        if not np.allclose(mags, mags[0]):
            raise ValueError(
                "Differential (non-coherent) detection needs a constant-modulus "
                "constellation such as PSK; this one has points of different magnitude."
            )

    def _slice_phase(self, z):
        """Closest constellation point by phase: argmax Re(conj(table) * z). Safe when z = 0."""
        z = np.asarray(z, dtype=complex)
        return np.argmax(np.real(np.conj(self.table) * z[..., None]), axis=-1)

    def differential_detect(self, y):
        """
        Differential (DPSK) detection for a transmitter that used differential_encode().

        y : (T+1,) received samples, or (Nr, T+1) for Nr receive antennas. The
            first sample is the reference. The detector never sees h or N0: with a
            channel that is constant over two consecutive samples,

                y_t * conj(y_{t-1}) = |h|^2 * sqrt(A) * s_t + noise,

        so the phase of the product gives the symbol. With several receive
        antennas the products are summed over the antennas (receive diversity)
        before slicing. Returns T symbol indices (not bits).

        The fading must stay (almost) constant between neighbouring samples,
        e.g. MIMOLTapChannel with L = 1. It does not work with Rayleigh_Channel,
        which draws a new gain for every sample.
        """
        self._require_psk()
        y = np.atleast_2d(np.asarray(y, dtype=complex))
        y = np.vstack([self.waveform.demodulate(row) for row in y])
        if y.shape[1] < 2:
            raise ValueError("Need the reference sample plus at least one data sample.")
        z = np.sum(y[:, 1:] * np.conj(y[:, :-1]), axis=0)
        return self._slice_phase(z)

    def decode_differential(self, y):
        """Differential detection followed by the bit lookup. Returns a 1-D bit array."""
        return self.bit_table[self.differential_detect(y)].ravel()

    def differential_alamouti_detect(self, Y):
        """
        Non-coherent detection for 2 transmit antennas, for a transmitter that used
        differential_alamouti_encode() (Tarokh-Jafarkhani differential STBC).

        Y : (Nr, 2*(P+1)) received matrix, first block = reference. Block t obeys
        Y_t = Y_{t-1} B(s1, s2) + noise, where B = [[s1, -s2*], [s2, s1*]], so the
        pair (s1, s2) is found from two consecutive blocks without knowing H:

            s1_hat = sum_r [conj(p1) q1 + p2 conj(q2)] / E
            s2_hat = sum_r [conj(p2) q1 - p1 conj(q2)] / E

        with p1, p2 the two columns of the previous block, q1, q2 the current ones
        and E = sum_r (|p1|^2 + |p2|^2). This is Alamouti combining where the
        previous block plays the role of the channel. Returns 2P symbol indices.
        """
        self._require_psk()
        Y = np.atleast_2d(np.asarray(Y, dtype=complex))
        Y = np.vstack([self.waveform.demodulate(row) for row in Y])
        if Y.shape[1] % 2 != 0 or Y.shape[1] < 4:
            raise ValueError("Y needs an even number of columns: reference block plus at least one data block.")

        c1, c2 = Y[:, 0::2], Y[:, 1::2]                 # first / second column of every block
        p1, p2, q1, q2 = c1[:, :-1], c2[:, :-1], c1[:, 1:], c2[:, 1:]
        E = np.maximum(np.sum(np.abs(p1) ** 2 + np.abs(p2) ** 2, axis=0), 1e-300)
        s1 = np.sum(np.conj(p1) * q1 + p2 * np.conj(q2), axis=0) / E
        s2 = np.sum(np.conj(p2) * q1 - p1 * np.conj(q2), axis=0) / E

        # s_i = a_i / sqrt(2) with a_i a unit-modulus PSK point; bring back to the constellation scale
        scale = np.sqrt(2 * self.constellation.A)
        idx = np.empty(2 * s1.size, dtype=int)
        idx[0::2] = self._slice_phase(scale * s1)
        idx[1::2] = self._slice_phase(scale * s2)
        return idx

    def decode_differential_alamouti(self, Y):
        """Differential Alamouti detection followed by the bit lookup. Returns a 1-D bit array."""
        return self.bit_table[self.differential_alamouti_detect(Y)].ravel()

    # ---------- non-coherent: energy detection (OOK / amplitude constellations) ----------
    def energy_detect(self, y, N0=None, fading_var=1.0):
        """
        Energy (envelope) detector for constellations whose points differ in
        MAGNITUDE only, e.g. OOK = Constellation("OOK", mapping={"0": 0, "1": 1}) or
        several amplitude levels. The phase is never used, so no channel phase,
        no pilots and no h are needed, and it also works with the per-sample
        gains of Rayleigh_Channel.

        y : (T,) or (Nr, T) received samples. The energy is summed over the Nr antennas.

        fading_var = E|h|^2 (default 1, Rayleigh like the project channels), N0 required:
            the energy under symbol i is exponential (Gamma with Nr terms) with mean
            m_i = N0 + fading_var*|s_i|^2, so the ML rule picks the i that maximises
            -Nr*ln(m_i) - E/m_i  (for OOK this is a single threshold on E).
        fading_var=None (plain AWGN, h = 1): pick the level whose energy |s_i|^2 + N0
            is closest to the measured average energy; N0 is not needed.

        Caveat: with fading the gain scales the energy, so a constellation with more than
        two levels gets confused even without noise. Use OOK under fading; amplitude
        levels (ASK) are only reliable on AWGN (fading_var=None).

        Returns symbol indices (not bits). Raises ValueError if two constellation
        points share the same magnitude (e.g. PAM/PSK/QAM symmetric points).
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
        E = np.sum(np.abs(y) ** 2, axis=0)                       # (T,)

        if fading_var is None:
            n0 = 0.0 if N0 is None else N0
            return np.argmin(np.abs(E[:, None] / Nr - (levels[None, :] + n0)), axis=1)

        if N0 is None or N0 <= 0:
            raise ValueError("Fading ML energy detection needs N0 > 0.")
        if fading_var <= 0:
            raise ValueError("fading_var must be positive (or None for plain AWGN).")
        m = N0 + fading_var * levels                              # (M,)
        return np.argmax(-Nr * np.log(m)[None, :] - E[:, None] / m[None, :], axis=1)

    def decode_energy(self, y, N0=None, fading_var=1.0):
        """Energy detection followed by the bit lookup. Returns a 1-D bit array."""
        return self.bit_table[self.energy_detect(y, N0, fading_var)].ravel()

    # ---------- non-coherent: orthogonal M-FSK ----------
    def fsk_detect(self, Y):
        """
        Non-coherent detection of orthogonal M-FSK (M = constellation.M tones; the
        symbol index is the tone index, the constellation points themselves are not used).

        Y : (M, T) correlator outputs, row m = tone m, or (Nr, M, T) for Nr receive
        antennas. In every branch y_m = h*x_m + n_m with x_m = sqrt(A) for the sent
        tone and 0 otherwise, so the phase of h is irrelevant: pick the tone with the
        largest energy |y_m|^2 (summed over antennas). This is the ML rule for
        Rayleigh fading and for AWGN with unknown phase, and needs neither h nor N0.
        Returns T symbol indices.
        """
        Y = np.asarray(Y, dtype=complex)
        if Y.ndim == 2:
            Y = Y[None]
        if Y.ndim != 3 or Y.shape[1] != self.M:
            raise ValueError("Y must have shape (M, T) or (Nr, M, T) with M = constellation.M tones.")
        return np.argmax(np.sum(np.abs(Y) ** 2, axis=0), axis=0)

    def decode_fsk(self, Y):
        """Non-coherent FSK detection followed by the bit lookup. Returns a 1-D bit array."""
        return self.bit_table[self.fsk_detect(Y)].ravel()

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


def pilot_matrix(n_tx, n_pilots=None, Es=1.0, power_split=True):
    """
    Orthogonal pilot matrix Xp of shape (n_tx, n_pilots), n_pilots >= n_tx
    (default n_pilots = n_tx). Rows are DFT sequences, so every pilot has
    constant modulus and Xp Xp^H is a multiple of the identity, which makes the
    least-squares estimate as clean as it can be.

    Energy follows CodingScheme: total energy per channel use is Es (each
    antenna gets Es/n_tx) when power_split=True, otherwise Es per antenna.
    The transmitter and receiver must build the same matrix, so both can call
    this function (it is deterministic).
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
    Differential PSK encoder (transmit side of Receiver.differential_detect).

    symbols : constant-modulus constellation symbols (|s|^2 = A, e.g. PSK from
              Transmitter.run with the SC waveform and no cyclic prefix).
    Returns T+1 symbols: a reference sqrt(A) followed by d_t = d_{t-1} * s_t / sqrt(A),
    i.e. the information is in the phase change between neighbouring samples.
    Feed the result to the channel like ordinary symbols.
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
    Differential Alamouti encoder (transmit side of
    Receiver.differential_alamouti_detect). Two antennas, shape (2, 2*(P+1)).

    symbols : 2P constant-modulus PSK symbols with |s|^2 = Es, taken in pairs (u1, u2).
    Each pair becomes s = (u1, u2) / sqrt(2*Es), and the transmitted blocks are
    X_t = scale * U_t with U_t = U_{t-1} * [[s1, -s2*], [s2, s1*]]. The first block
    is a fixed unitary reference. U_t stays unitary, so every channel use carries
    energy Es in total (power_split=True, like CodingScheme) or Es per antenna
    (power_split=False).
    """
    u = np.asarray(symbols, dtype=complex).ravel()
    if Es <= 0:
        raise ValueError("Es must be positive.")
    if u.size == 0 or u.size % 2 != 0:
        raise ValueError("Need a non-empty even number of symbols.")
    if not np.allclose(np.abs(u) ** 2, Es):
        raise ValueError("Differential Alamouti needs constant-modulus symbols with |s|^2 = Es (PSK).")

    a = u / np.sqrt(Es) / np.sqrt(2)                      # |a1|^2 + |a2|^2 = 1
    P = a.size // 2
    r = 1 / np.sqrt(2)
    U = np.array([[r, -r], [r, r]], dtype=complex)        # reference block (Alamouti form, unitary)
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
    Orthogonal M-FSK in the correlator domain (transmit side of Receiver.fsk_detect).

    indices : symbol indices 0..M-1, e.g. Receiver.bits_to_indices(bits).
    Returns an (M, T) matrix with sqrt(Es) in row indices[t] of column t and 0
    elsewhere: one tone per channel use, Es energy per symbol. Pass every row
    through the SAME channel gain (h*X + noise), since all tones share one fading
    coefficient; use Nr of them for receive diversity.
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
