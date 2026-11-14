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
    (V-BLAST), Alamouti combining and pilot-based channel estimation,
    Viterbi MLSE for single-carrier ISI channels, SISO or MIMO (mlse_detect, mlse_detect_mimo) with
    reduced-state variants (M-algorithm, K-best sphere candidates, channel-shortening prefilter),
    time-domain ZF / MMSE linear and decision-feedback equalizers for single-carrier ISI channels
    (equalize_sc, detect_sc), per-subcarrier MIMO-OFDM detection (ofdm_mimo_detect), plus
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

    # ---------- Viterbi / reduced-state MLSE for single-carrier ISI channels ----------
    # Model: Y[:, n] = sum_j H[:, :, j] @ x[:, n-j] + noise, with one symbol per transmit antenna in
    # x[:, n]. The Nt symbols sent in one step are treated as ONE joint symbol from an alphabet of
    # size Mv = M^Nt (antenna i is digit i of the joint index, base M), so SISO is just Nt = 1.

    def _joint_values(self, Nt):
        """(Mv, Nt) symbol values of every joint symbol index."""
        digits = (np.arange(self.M ** Nt)[:, None] // self.M ** np.arange(Nt)[None, :]) % self.M
        return self.table[digits]

    # ----- channel-shortening prefilter (MMSE-DFE style) -----
    def design_prefilter(self, H, N0, n_taps, n_ff=None, delay=None, power_split=True):
        """
        Design a prefilter that shortens the channel to n_taps taps for the trellis.

        H has shape (Nr, Nt, L) (as returned by MIMOLTapChannel; (Nr, Nt) or a 1-D h are also
        accepted). The filter W (n_ff taps, Nt x Nr each, delayed by `delay` samples) and the
        monic target B (n_taps matrices, Nt x Nt) minimise E|x[m] - W*y - feedback|^2, i.e. the
        MMSE-DFE solution (Falconer-Magee prefilter): W*y = sum_j B_j x[m-j] + e[m]. The residual e
        is then whitened across antennas, so the Viterbi metric is a plain Euclidean distance on a
        channel with only n_taps taps (trellis size M^(Nt*(n_taps-1)) instead of M^(Nt*(L-1))).
        Needs the noise variance N0 > 0. delay=None picks the delay with the smallest error.

        Returns dict(W=(n_ff, Nt, Nr), B=(n_taps, Nt, Nt), delay=int, error_cov=(Nt, Nt)).
        """
        H = np.asarray(H, dtype=complex)
        if H.ndim == 1:
            H = H[None, None, :]
        elif H.ndim == 2:
            H = H[:, :, None]
        if H.ndim != 3:
            raise ValueError("H must have shape (Nr, Nt, L).")
        Nr, Nt, L = H.shape
        if not N0 > 0:
            raise ValueError("The prefilter design needs N0 > 0.")
        Lt = int(n_taps)
        if Lt < 1:
            raise ValueError("n_taps must be a positive integer.")
        Nw = max(2 * L, 8) if n_ff is None else int(n_ff)
        if Nw < 1 or Nw + L - 1 < Lt:
            raise ValueError("n_ff is too small for this many target taps.")
        if power_split:
            H = H / np.sqrt(Nt)
        A = self.constellation.A

        nx = Nw + L - 1                               # symbols seen by the prefilter window
        Hs = np.zeros((Nr * Nw, Nt * nx), dtype=complex)
        for k in range(Nw):                            # y[m+d-k] = sum_l H_l x[m+d-k-l]
            for l in range(L):
                Hs[k * Nr:(k + 1) * Nr, (k + l) * Nt:(k + l + 1) * Nt] = H[:, :, l]
        dim_y, dim_f = Nr * Nw, Nt * (Lt - 1)

        best = None
        for d in ([int(delay)] if delay is not None else range(nx - Lt + 1)):
            if not 0 <= d <= nx - Lt:
                raise ValueError("delay is out of range for n_ff and n_taps.")
            S0 = np.zeros((Nt, Nt * nx), dtype=complex)
            S0[:, d * Nt:(d + 1) * Nt] = np.eye(Nt)
            Sf = np.zeros((dim_f, Nt * nx), dtype=complex)
            for j in range(1, Lt):
                Sf[(j - 1) * Nt:j * Nt, (d + j) * Nt:(d + j + 1) * Nt] = np.eye(Nt)
            Om = np.vstack([Hs, Sf])                   # phi = [y window; fed-back symbols] = Om x + [v; 0]
            Rpp = A * Om @ Om.conj().T
            Rpp[:dim_y, :dim_y] += N0 * np.eye(dim_y)
            Rxp = A * S0 @ Om.conj().T
            theta = np.linalg.solve(Rpp, Rxp.conj().T).conj().T
            Se = A * np.eye(Nt) - theta @ Rxp.conj().T
            Se = (Se + Se.conj().T) / 2 + 1e-12 * A * np.eye(Nt)
            score = np.linalg.slogdet(Se)[1]
            if best is None or score < best[0]:
                best = (score, d, theta, Se)
        _, d, theta, Se = best

        Cinv = np.linalg.inv(np.linalg.cholesky(Se))   # whitens the residual: Cinv Se Cinv^H = I
        W = np.stack([Cinv @ theta[:, k * Nr:(k + 1) * Nr] for k in range(Nw)])
        B = np.stack([Cinv] + [-Cinv @ theta[:, dim_y + (j - 1) * Nt:dim_y + j * Nt]
                               for j in range(1, Lt)])
        return {"W": W, "B": B, "delay": d, "error_cov": Se}

    @staticmethod
    def _apply_prefilter(Y, W, delay, T):
        """z[m] = sum_k W_k y[m + delay - k] for m = 0..T-1 (zeros outside the received block)."""
        Nw, Nt, Nr = W.shape
        Yp = np.concatenate([np.zeros((Nr, Nw), dtype=complex), Y,
                             np.zeros((Nr, delay), dtype=complex)], axis=1)
        Z = np.zeros((Nt, T), dtype=complex)
        for k in range(Nw):
            s = delay - k + Nw
            Z += W[k] @ Yp[:, s:s + T]
        return Z

    # ----- K-best sphere decoder (candidate generator) -----
    def _ksphere_prepare(self, G0):
        """QR of G0 (Nrx x Nt) with the strongest column decided first. Needs Nrx >= Nt."""
        if G0.shape[0] < G0.shape[1]:
            raise ValueError("Sphere-decoder candidates need Nr >= Nt.")
        perm = np.argsort(np.linalg.norm(G0, axis=0))
        Q, R = np.linalg.qr(G0[:, perm])
        return perm, Q, R

    def _ksphere(self, r, prep, K):
        """
        For every row r[p] (Nrx,) return the K joint symbols x with the smallest ||r - G0 x||^2
        as found by breadth-first (K-best) sphere decoding: (P, K') joint indices, K' <= K.
        The Nt antenna levels of the triangular system are searched top-down, keeping the K best
        partial paths per level. Cost ~ P*K*M*Nt instead of P*M^Nt.
        """
        perm, Q, R = prep
        Nt, M, P = R.shape[0], self.M, r.shape[0]
        y = r @ Q.conj()                                   # y = Q^H r, shape (P, Nt)
        pd = np.zeros((P, 1))
        sel = np.zeros((P, 1, 0), dtype=int)
        for lev in range(Nt - 1, -1, -1):
            res = np.broadcast_to(y[:, lev, None], (P, 1))
            if sel.shape[2]:
                res = y[:, lev, None] - self.table[sel] @ R[lev, Nt - 1:lev:-1]
            d = np.abs(res[:, :, None] - R[lev, lev] * self.table[None, None, :]) ** 2
            flat = (pd[:, :, None] + d).reshape(P, -1)
            k = min(K, flat.shape[1])
            if k < flat.shape[1]:
                idx = np.argpartition(flat, k - 1, axis=1)[:, :k]
            else:
                idx = np.broadcast_to(np.arange(k), (P, k))
            pd = np.take_along_axis(flat, idx, axis=1)
            parent, new = idx // M, idx % M
            sel = np.concatenate([np.take_along_axis(sel, parent[:, :, None], axis=1),
                                  new[:, :, None]], axis=2) if sel.shape[2] else new[:, :, None]
        v = np.zeros(sel.shape[:2], dtype=np.int64)
        for t in range(Nt):                                # decided order t <-> level Nt-1-t
            v += sel[:, :, t] * M ** int(perm[Nt - 1 - t])
        return v

    # ----- detectors -----
    def _mlse_flat(self, Z, G0, sv, n_cand):
        """Memoryless (L = 1) joint ML detection, exact or with K-best sphere candidates."""
        Mv, Nrx, T = sv.shape[0], Z.shape[0], Z.shape[1]
        cont = sv @ G0.T                                   # (Mv, Nrx) noiseless output per joint symbol
        En = np.sum(np.abs(cont) ** 2, axis=1)
        prep = None if (n_cand is None or n_cand >= Mv) else self._ksphere_prepare(G0)
        out = np.empty(T, dtype=np.int64)
        step = max(1, (1 << 21) // (Mv if prep is None else n_cand * Nrx))
        for a in range(0, T, step):
            r = Z[:, a:a + step].T                          # (c, Nrx)
            if prep is None:
                out[a:a + step] = np.argmin(En[None, :] - 2 * (r.conj() @ cont.T).real, axis=1)
            else:
                cand = self._ksphere(r, prep, n_cand)
                dist = np.sum(np.abs(r[:, None, :] - cont[cand]) ** 2, axis=2)
                out[a:a + step] = np.take_along_axis(cand, dist.argmin(1)[:, None], 1)[:, 0]
        return out

    @staticmethod
    def _isi_table(contrib, Mv, L, jlo, jhi):
        """
        Noiseless output of every trellis transition, shape (K, Mv, Mv, Nrx) with K = Mv^(L-2).
        State after step n: s = v[n-1] + Mv*v[n-2] + ... (joint indices, newest = least significant).
        A transition into s' = q*Mv + u (u = v[n]) comes from s = q + K*d, d = v[n-L+1] (the oldest
        joint symbol, which leaves the state). Axes: (q, u, d). Only taps jlo..jhi are used; others
        point before the first or after the last real symbol and are zero.
        """
        K = Mv ** (L - 2)
        q = np.arange(K)
        E = np.zeros((K, Mv, Mv, contrib.shape[2]), dtype=complex)
        for j in range(jlo, jhi + 1):
            if j == 0:
                E += contrib[0][None, :, None, :]
            elif j == L - 1:
                E += contrib[j][None, None, :, :]
            else:
                E += contrib[j][(q // Mv ** (j - 1)) % Mv][:, None, None, :]
        return E

    def _mlse_viterbi(self, Z, contrib, Mv, L, T, depth, block):
        """Full-state Viterbi with a bounded sliding-window traceback."""
        Nrx, n_steps = Z.shape
        S, K = Mv ** (L - 1), Mv ** (L - 2)
        if depth == "full":
            W = D = 0
        else:
            D = int(depth)
            W = D + (D if block is None else int(block))
        if W == 0 or W > n_steps:
            W, D = n_steps, 0
        B = W - D
        buf = np.empty((W, S), dtype=np.min_scalar_type(Mv - 1))   # survivor decisions
        out = np.empty(n_steps, dtype=np.min_scalar_type(Mv - 1))
        zc = np.ascontiguousarray(np.concatenate([Z.real, Z.imag], axis=0).T)   # (n_steps, 2*Nrx)
        pm = np.zeros(S)

        def table(jlo, jhi):
            E = self._isi_table(contrib, Mv, L, jlo, jhi)
            return (np.concatenate([E.real, E.imag], axis=-1).reshape(-1, 2 * Nrx),
                    (E.real ** 2 + E.imag ** 2).sum(-1).reshape(-1))

        mid = table(0, L - 1)
        filled = emitted = 0

        def traceback(rows):
            s = int(pm.argmin())
            syms = np.empty(rows, dtype=out.dtype)
            for r in range(rows - 1, -1, -1):
                syms[r] = s % Mv
                s = s // Mv + K * int(buf[r, s])
            return syms

        for n in range(n_steps):
            jlo, jhi = max(0, n - T + 1), min(L - 1, n)
            Ecat, En = mid if (jlo, jhi) == (0, L - 1) else table(jlo, jhi)
            cand = (En - 2 * (Ecat @ zc[n])).reshape(K, Mv, Mv)   # |z|^2 dropped: same for all branches
            cand += pm.reshape(Mv, K).T[:, None, :]
            buf[filled] = cand.argmin(axis=2).reshape(S)
            pm = cand.min(axis=2).reshape(S)
            pm -= pm.min()
            filled += 1
            if filled == W and n + 1 < n_steps:
                out[emitted:emitted + B] = traceback(W)[:B]
                emitted += B
                buf[:D] = buf[B:W]
                filled = D
        out[emitted:emitted + filled] = traceback(filled)
        return out[:T].astype(np.int64)

    def _mlse_malg(self, Z, G, sv, contrib, Mv, L, T, depth, block, n_keep, n_cand):
        """
        M-algorithm: keep only the n_keep best paths, merging paths that share the same last L-1
        joint symbols. n_cand (optional) expands each path with only its n_cand best joint symbols,
        found by the K-best sphere decoder, instead of all Mv. Memory is bounded by a sliding window:
        (depth + block) x n_keep (parent, symbol) records.
        """
        Nrx, n_steps = Z.shape
        if depth == "full":
            W = D = 0
        else:
            D = int(depth)
            W = D + (D if block is None else int(block))
        if W == 0 or W > n_steps:
            W, D = n_steps, 0
        B = W - D
        Kd = Mv ** (L - 2)
        merge = Mv ** (L - 1) < 2 ** 62
        use_sphere = n_cand is not None and n_cand < Mv
        prep = self._ksphere_prepare(G[0]) if use_sphere else None
        par = np.zeros((W, n_keep), dtype=np.min_scalar_type(n_keep))
        sym = np.zeros((W, n_keep), dtype=np.min_scalar_type(Mv - 1))
        out = np.empty(n_steps, dtype=np.min_scalar_type(Mv - 1))
        pm = np.zeros(1)
        hist = np.zeros((1, L - 1), dtype=np.int64)       # hist[:, j-1] = joint index of v[n-j]
        key = np.zeros(1, dtype=np.int64)
        filled = emitted = 0

        def traceback(rows):
            k = 0                                          # survivors are kept sorted: 0 is the best
            syms = np.empty(rows, dtype=out.dtype)
            for r in range(rows - 1, -1, -1):
                syms[r] = sym[r, k]
                k = int(par[r, k])
            return syms

        for n in range(n_steps):
            jlo, jhi = max(0, n - T + 1), min(L - 1, n)
            P = pm.size
            r = np.broadcast_to(Z[:, n], (P, Nrx)).copy()
            for j in range(max(1, jlo), jhi + 1):          # ISI from each path's own history
                r -= contrib[j][hist[:, j - 1]]
            if jlo == 0:
                if use_sphere:
                    cands = self._ksphere(r, prep, n_cand)             # (P, Kc)
                    dd = r[:, None, :] - contrib[0][cands]
                else:
                    cands = np.broadcast_to(np.arange(Mv), (P, Mv))
                    dd = r[:, None, :] - contrib[0][None, :, :]
            else:                                          # past the last symbol: no new input
                cands = np.zeros((P, 1), dtype=np.int64)
                dd = r[:, None, :]
            Kc = cands.shape[1]
            tot = (pm[:, None] + (dd.real ** 2 + dd.imag ** 2).sum(-1)).ravel()
            parent = np.repeat(np.arange(P), Kc)
            v = np.ascontiguousarray(cands).ravel()
            if merge:
                nk = (key[parent] % Kd) * Mv + v
                order = np.argsort(tot, kind="stable")
                first = np.unique(nk[order], return_index=True)[1]
                sel = order[np.sort(first)][:n_keep]
            else:
                sel = np.argsort(tot, kind="stable")[:n_keep]
            pm = tot[sel]
            pm = pm - pm[0]
            P2 = sel.size
            par[filled, :P2], sym[filled, :P2] = parent[sel], v[sel]
            new_hist = np.empty((P2, L - 1), dtype=np.int64)
            new_hist[:, 0] = v[sel]
            new_hist[:, 1:] = hist[parent[sel], :L - 2]
            hist = new_hist
            if merge:
                key = nk[sel]
            filled += 1
            if filled == W and n + 1 < n_steps:
                out[emitted:emitted + B] = traceback(W)[:B]
                emitted += B
                par[:D], sym[:D] = par[B:W].copy(), sym[B:W].copy()
                filled = D
        out[emitted:emitted + filled] = traceback(filled)
        return out[:T].astype(np.int64)

    def mlse_detect_mimo(self, Y, H, tail=False, power_split=True, method="viterbi",
                         depth=None, block=None, n_keep=None, n_cand=None,
                         shorten=None, N0=None, n_ff=None, delay=None, max_states=1 << 16):
        """
        Sequence detection for Y[:, n] = sum_j H[:, :, j] x[:, n-j] + noise (V-BLAST symbols over a
        frequency-selective MIMO channel; SISO is Nr = Nt = 1). Single carrier, no cyclic prefix,
        known H of shape (Nr, Nt, L) as returned by MIMOLTapChannel, white noise.

        Y     : (Nr, T) received samples (tail=False) or (Nr, T + L - 1) (tail=True).
        method: "viterbi"  exact ML over a trellis of M^(Nt*(L-1)) states (guarded by max_states);
                "malg"     M-algorithm: only the n_keep best paths survive (default 16), merging
                           paths with identical last L-1 symbols; approximate ML.
        n_cand: with method="malg" (or a memoryless channel) expand each path with only its n_cand
                best joint symbols from a K-best sphere decoder instead of all M^Nt. Needs Nr >= Nt.
        shorten: if given, first apply an MMSE-DFE-style prefilter (design_prefilter, needs N0) so
                the trellis only needs `shorten` taps; n_ff / delay tune that filter.
        depth, block: sliding-window traceback in symbols (default 8x the trellis memory); depth="full"
                keeps every decision (exact, memory grows with T).
        power_split : H is divided by sqrt(Nt), as the other MIMO methods do for the Coding_Schemes.

        Returns the (Nt, T) array of detected constellation indices.
        """
        if self.waveform.name != "SC" or self.waveform.cyclic_prefix:
            raise ValueError("MLSE needs single-carrier transmission without a cyclic prefix.")
        method = str(method).lower()
        if method not in ("viterbi", "malg"):
            raise ValueError("method must be 'viterbi' or 'malg'.")
        H = np.asarray(H, dtype=complex)
        if H.ndim == 1:
            H = H[None, None, :]
        elif H.ndim == 2:
            H = H[:, :, None]
        Y = np.atleast_2d(np.asarray(Y, dtype=complex))
        if H.ndim != 3 or Y.ndim != 2 or Y.shape[0] != H.shape[0] or Y.size == 0:
            raise ValueError("Y must be (Nr, samples) and H must be (Nr, Nt, L) with the same Nr.")
        if not (np.all(np.isfinite(H)) and np.all(np.isfinite(Y))):
            raise ValueError("H and Y must be finite.")
        Nr, Nt, L = H.shape
        T = Y.shape[1] - (L - 1) if tail else Y.shape[1]
        if T < 1:
            raise ValueError("Y is too short for this many taps.")
        for name, val in (("n_keep", n_keep), ("n_cand", n_cand)):
            if val is not None and (int(val) != val or val < 1):
                raise ValueError(f"{name} must be a positive integer.")
        Mv = self.M ** Nt
        Heff = H / np.sqrt(Nt) if power_split else H

        if shorten is None:
            G, Z, Leff, T_steps = np.transpose(Heff, (2, 0, 1)), Y, L, Y.shape[1]
        else:
            if N0 is None:
                raise ValueError("shorten needs N0 to design the prefilter.")
            pf = self.design_prefilter(Heff, N0, shorten, n_ff, delay, power_split=False)
            Z = self._apply_prefilter(Y, pf["W"], pf["delay"], T)
            G, Leff, T_steps = pf["B"], int(shorten), T

        sv = self._joint_values(Nt)
        if Leff == 1:
            v = self._mlse_flat(Z[:, :T], G[0], sv, n_cand)
        else:
            if depth is None:
                depth = 8 * Leff
            elif depth != "full" and (int(depth) != depth or depth < Leff):
                raise ValueError("depth must be an integer >= the trellis memory, or 'full'.")
            if block is not None and (int(block) != block or block < 1):
                raise ValueError("block must be a positive integer.")
            contrib = np.einsum("jrt,vt->jvr", G, sv)      # (L, Mv, Nrx): output of tap j per joint symbol
            if method == "viterbi":
                if n_cand is not None or n_keep is not None:
                    raise ValueError("n_keep / n_cand only apply to method='malg'.")
                S = Mv ** (Leff - 1)
                if S > max_states or S * Mv * Z.shape[0] > 1 << 22:
                    raise ValueError(
                        f"The trellis has M^(Nt*(L-1)) = {S} states with {Mv} branches each, which is "
                        "too large for full Viterbi (raise max_states, or use method='malg', "
                        "n_cand and/or shorten)."
                    )
                v = self._mlse_viterbi(Z[:, :T_steps], contrib, Mv, Leff, T, depth, block)
            else:
                v = self._mlse_malg(Z[:, :T_steps], G, sv, contrib, Mv, Leff, T, depth, block,
                                    16 if n_keep is None else int(n_keep),
                                    None if n_cand is None else int(n_cand))
        return (v[None, :] // self.M ** np.arange(Nt)[:, None]) % self.M

    def decode_mlse_mimo(self, Y, H, **kwargs):
        """mlse_detect_mimo followed by the bit lookup (bits in V-BLAST stream order)."""
        return self.bit_table[self._stream(self.mlse_detect_mimo(Y, H, **kwargs))].ravel()

    # ---------- SISO wrappers ----------
    def mlse_detect(self, y, h, tail=False, depth=None, block=None, max_states=1 << 16, **kwargs):
        """
        Viterbi MLSE for y[n] = sum_j h[j] x[n-j] + noise (SISO, single carrier, no CP, known taps).
        len(y) = T (tail dropped, as in flow_test) or T + L - 1 with tail=True. depth / block set the
        sliding-window traceback (default 8*L); depth="full" is exact but needs T x M^(L-1) bytes.
        Extra keywords (method="malg", n_keep, n_cand, shorten, N0, ...) are passed on to
        mlse_detect_mimo. Returns the (T,) array of constellation indices.
        """
        y = np.asarray(y, dtype=complex).ravel()
        h = np.asarray(h, dtype=complex).ravel()
        if h.size == 0 or y.size == 0:
            raise ValueError("h and y must be non-empty.")
        return self.mlse_detect_mimo(y[None, :], h[None, None, :], tail=tail, power_split=False,
                                     depth=depth, block=block, max_states=max_states, **kwargs)[0]

    def decode_mlse(self, y, h, **kwargs):
        """Viterbi MLSE equalization followed by the bit lookup."""
        return self.bit_table[self.mlse_detect(y, h, **kwargs)].ravel()

    # ---------- Time-domain equalizers for single-carrier ISI channels ----------
    # Model: Y[:, n] = sum_j H[:, :, j] x[:, n-j] + noise. Zero forcing (ZF) or MMSE, linear or
    # decision feedback (DFE), SISO or MIMO. A filter window of n_ff taps looks at the received
    # samples y[m+d-k], k = 0..n_ff-1 (d = look-ahead, chosen per transmit antenna); a DFE also
    # subtracts the channel's response to the n_fb previous DECISIONS and, optionally, to the decisions
    # already made for the other antennas at the same time step (spatial/V-BLAST-style feedback).

    @staticmethod
    def _stack_channel(H, Nw):
        """
        Block-Toeplitz Hs of shape (Nr*Nw, Nt*(Nw+L-1)) with  y_window = Hs @ x_window  (noise-free).
        y_window[k*Nr:(k+1)*Nr] = y[m+d-k];  x_window block i (Nt entries) = x[m+d-i].
        """
        Nr, Nt, L = H.shape
        nx = Nw + L - 1
        Hs = np.zeros((Nr * Nw, Nt * nx), dtype=complex)
        for k in range(Nw):
            for l in range(L):
                Hs[k * Nr:(k + 1) * Nr, (k + l) * Nt:(k + l + 1) * Nt] = H[:, :, l]
        return Hs, nx

    def design_equalizer(self, H, N0=None, kind="mmse", structure="linear", n_ff=None, n_fb=None,
                         delay=None, spatial=None, order=None, power_split=True):
        """
        Design a time-domain equalizer for the channel H of shape (Nr, Nt, L) (a 1-D h is SISO).

        kind     : "mmse" (needs N0) or "zf". ZF removes all ISI it can: the least-squares / minimum-
                   norm FIR solution, so it is exact only when the filter has enough degrees of
                   freedom (SISO DFE with n_fb >= L-1; MIMO needs Nr > Nt for a linear ZF).
        structure: "linear" or "dfe". A DFE feeds back past decisions: the feedforward filter is
                   designed so the response to the n_fb previous symbols (and to already-decided
                   antennas, if spatial) can be cancelled instead of suppressed.
        n_ff, n_fb: feedforward taps (default max(4L, 8)) and feedback taps (default L-1 for a DFE).
        delay    : look-ahead in samples; None searches for the best one per antenna.
        spatial  : DFE only (default: True for Nt > 1). Also cancel the same-time decisions of the
                   antennas detected earlier; order=None chooses the order greedily (best first),
                   or pass a permutation of range(Nt).
        N0       : noise variance. For kind="zf" it is only used to pick the delay/order.

        Note: a ZF-DFE has no spare degrees of freedom (its constraints exactly determine the
        feedforward filter), so on channels with zeros outside the unit circle it needs a long
        n_ff and still enhances noise badly; on a minimum-phase channel it reaches the AWGN
        bound with feedback taps -h[1:]. The MMSE-DFE does not have this weakness.

        Returns dict(W=(n_ff, Nt, Nr) feedforward taps, delay=(Nt,), Fb=(Nt, n_fb, Nt) feedback
        taps, Sm=(Nt, Nt) spatial feedback, gain=(Nt,) gain on the desired symbol (for unbiasing),
        order, score=(Nt,)). Detection with these is done by equalize_sc / detect_sc.
        """
        kind, structure = str(kind).lower(), str(structure).lower()
        if kind not in ("zf", "mmse") or structure not in ("linear", "dfe"):
            raise ValueError("kind must be 'zf' or 'mmse' and structure 'linear' or 'dfe'.")
        H = np.asarray(H, dtype=complex)
        if H.ndim == 1:
            H = H[None, None, :]
        elif H.ndim == 2:
            H = H[:, :, None]
        if H.ndim != 3:
            raise ValueError("H must have shape (Nr, Nt, L).")
        Nr, Nt, L = H.shape
        A = self.constellation.A
        if kind == "mmse" and (N0 is None or N0 < 0):
            raise ValueError("MMSE equalization needs N0 >= 0.")
        if kind == "zf" and Nr < Nt:
            raise ValueError("Zero forcing needs Nr >= Nt.")
        n0 = 0.01 * A if N0 is None else float(N0)
        dfe = structure == "dfe"
        n_fb = (L - 1 if dfe else 0) if n_fb is None else int(n_fb)
        if n_fb < 0 or (n_fb and not dfe):
            raise ValueError("n_fb must be a non-negative integer and is only used by a DFE.")
        if spatial is None:
            spatial = dfe and Nt > 1
        if spatial and not dfe:
            raise ValueError("Spatial feedback needs structure='dfe'.")
        Nw = max(4 * L, 8) if n_ff is None else int(n_ff)
        if Nw < 1:
            raise ValueError("n_ff must be a positive integer.")
        Hs, nx = self._stack_channel(H / np.sqrt(Nt) if power_split else H, Nw)
        if nx - n_fb < 1:
            raise ValueError("n_ff is too small for this many feedback taps.")
        delays = range(nx - n_fb) if delay is None else [int(delay)]
        if not all(0 <= d < nx - n_fb for d in delays):
            raise ValueError("delay is out of range for n_ff and n_fb.")
        dimY, ncol = Nr * Nw, Nt * nx
        G = Hs @ Hs.conj().T

        def solve(i, D):
            """Best (score, delay, w, f, gain) for antenna i, with antennas D already decided."""
            best = None
            for d in delays:
                tp = d * Nt + i
                P = [(d + j) * Nt + l for j in range(1, n_fb + 1) for l in range(Nt)]
                P += [d * Nt + l for l in D]
                ht = Hs[:, tp]
                if kind == "mmse":
                    nP = len(P)
                    Hp = Hs[:, P]
                    R = np.zeros((dimY + nP, dimY + nP), dtype=complex)
                    R[:dimY, :dimY] = A * G + n0 * np.eye(dimY)
                    R[:dimY, dimY:] = A * Hp
                    R[dimY:, :dimY] = A * Hp.conj().T
                    R[dimY:, dimY:] = A * np.eye(nP)
                    R += 1e-10 * A * np.eye(dimY + nP)
                    rhs = np.concatenate([A * ht, np.zeros(nP)])
                    theta = np.linalg.solve(R, rhs).conj()      # E[x|phi] = theta @ [y_window; fed-back x]
                    score = A - (theta @ rhs).real              # MMSE
                    w, f = theta[:dimY], theta[dimY:]
                else:
                    keep = np.ones(ncol, dtype=bool)
                    keep[P] = False
                    Hc = Hs[:, keep]
                    t = np.zeros(Hc.shape[1])
                    t[int(np.sum(keep[:tp]))] = 1.0
                    w = t @ np.linalg.pinv(Hc)                  # min-norm / LS: w Hc = t (unit at x[m])
                    res = w @ Hc - t
                    score = n0 * np.vdot(w, w).real + A * np.vdot(res, res).real
                    f = -(w @ Hs[:, P])                         # cancel the free (fed-back) columns
                if best is None or score < best[0]:
                    best = (score, d, w, f, w @ ht)
            return best

        chosen, D = {}, []
        if spatial and order is None:
            remaining = list(range(Nt))
            while remaining:
                cand = {i: solve(i, D) for i in remaining}
                i = min(cand, key=lambda c: cand[c][0])
                chosen[i] = (cand[i], list(D))
                D.append(i)
                remaining.remove(i)
            order = D
        else:
            order = list(range(Nt)) if order is None else [int(o) for o in order]
            if sorted(order) != list(range(Nt)):
                raise ValueError("order must be a permutation of range(Nt).")
            for i in order:
                chosen[i] = (solve(i, D if spatial else []), list(D) if spatial else [])
                if spatial:
                    D.append(i)

        W = np.zeros((Nw, Nt, Nr), dtype=complex)
        Fb = np.zeros((Nt, n_fb, Nt), dtype=complex)
        Sm = np.zeros((Nt, Nt), dtype=complex)
        dl = np.zeros(Nt, dtype=int)
        gain = np.zeros(Nt, dtype=complex)
        score = np.zeros(Nt)
        for i, ((sc, d, w, f, g), Di) in chosen.items():
            W[:, i, :] = w.reshape(Nw, Nr)
            if n_fb:
                Fb[i] = f[:n_fb * Nt].reshape(n_fb, Nt)
            for q, l in enumerate(Di):
                Sm[i, l] = f[n_fb * Nt + q]
            dl[i], gain[i], score[i] = d, g, sc
        return {"W": W, "delay": dl, "Fb": Fb, "Sm": Sm, "gain": gain, "order": list(order),
                "score": score, "kind": kind, "structure": structure, "spatial": bool(spatial)}

    def _run_equalizer(self, Y, des, T, unbiased):
        """Apply a designed equalizer to Y (Nr, samples): returns (soft (Nt, T), indices (Nt, T))."""
        W, dl = des["W"], des["delay"]
        Nw, Nt, Nr = W.shape
        Yp = np.concatenate([np.zeros((Nr, Nw), dtype=complex), Y,
                             np.zeros((Nr, int(dl.max())), dtype=complex)], axis=1)
        Zff = np.zeros((Nt, T), dtype=complex)
        for i in range(Nt):
            for k in range(Nw):
                s = int(dl[i]) - k + Nw
                Zff[i] += W[k, i, :] @ Yp[:, s:s + T]
        gain = des["gain"] if unbiased else np.ones(Nt)
        Fb, Sm = des["Fb"], des["Sm"]
        n_fb = Fb.shape[1]
        if n_fb == 0 and not des["spatial"]:
            soft = Zff / gain[:, None]
            return soft, self._slice(soft)

        soft = np.empty((Nt, T), dtype=complex)
        idx = np.empty((Nt, T), dtype=int)
        xh = np.zeros((Nt, T + n_fb), dtype=complex)      # decisions; column c is time c - n_fb
        for m in range(T):
            fb = (np.einsum("ijl,lj->i", Fb, xh[:, m:m + n_fb][:, ::-1]) if n_fb
                  else np.zeros(Nt, dtype=complex))
            for i in des["order"]:
                z = (Zff[i, m] + fb[i] + Sm[i] @ xh[:, m + n_fb]) / gain[i]
                k = int(np.argmin(np.abs(z - self.table) ** 2))
                soft[i, m], idx[i, m], xh[i, m + n_fb] = z, k, self.table[k]
        return soft, idx

    def _sc_equalizer(self, Y, H, N0, tail, unbiased, **design_kw):
        if self.waveform.name != "SC" or self.waveform.cyclic_prefix:
            raise ValueError("Time-domain equalizers need single-carrier transmission without a "
                             "cyclic prefix (use ofdm_mimo_detect for OFDM / SC with a CP).")
        siso = np.ndim(Y) == 1 and np.ndim(H) == 1
        H = np.asarray(H, dtype=complex)
        if H.ndim == 1:
            H = H[None, None, :]
        elif H.ndim == 2:
            H = H[:, :, None]
        Y = np.atleast_2d(np.asarray(Y, dtype=complex))
        if H.ndim != 3 or Y.ndim != 2 or Y.shape[0] != H.shape[0] or Y.size == 0:
            raise ValueError("Y must be (Nr, samples) and H must be (Nr, Nt, L) with the same Nr.")
        if not (np.all(np.isfinite(H)) and np.all(np.isfinite(Y))):
            raise ValueError("H and Y must be finite.")
        T = Y.shape[1] - (H.shape[2] - 1) if tail else Y.shape[1]
        if T < 1:
            raise ValueError("Y is too short for this many taps.")
        des = self.design_equalizer(H, N0, **design_kw)
        soft, idx = self._run_equalizer(Y, des, T, unbiased)
        return (soft[0], idx[0]) if siso else (soft, idx)

    def equalize_sc(self, Y, H, N0=None, tail=False, unbiased=True, **design_kw):
        """
        Equalized symbol estimates (before slicing) for a single-carrier ISI channel, shape (Nt, T)
        (1-D for SISO input). design_kw: kind, structure, n_ff, n_fb, delay, spatial, order,
        power_split (see design_equalizer). With a DFE these are the decision variables the slicer
        saw, i.e. they already include the feedback. unbiased=True divides out the MMSE bias so QAM
        can be sliced. With tail=False the last `delay` symbols miss their look-ahead samples and are
        less reliable (use tail=True when the full convolution output is available).
        """
        return self._sc_equalizer(Y, H, N0, tail, unbiased, **design_kw)[0]

    def detect_sc(self, Y, H, N0=None, tail=False, unbiased=True, **design_kw):
        """Constellation indices from a time-domain ZF/MMSE linear or decision-feedback equalizer."""
        return self._sc_equalizer(Y, H, N0, tail, unbiased, **design_kw)[1]

    def decode_sc(self, Y, H, N0=None, tail=False, unbiased=True, **design_kw):
        """Bits from detect_sc (V-BLAST stream order for MIMO)."""
        idx = self.detect_sc(Y, H, N0, tail, unbiased, **design_kw)
        return self.bit_table[idx if idx.ndim == 1 else self._stream(idx)].ravel()

    # ---------- Per-subcarrier MIMO-OFDM (and single-carrier frequency-domain) detection ----------
    def ofdm_mimo_detect(self, Y, H, N0=None, method="mmse", freq_domain=False, power_split=True,
                         ordered=True, return_soft=False):
        """
        Detect V-BLAST streams over a frequency-selective MIMO channel with the receiver's OFDM
        waveform (or single carrier with a cyclic prefix, i.e. SC-FDE): remove the CP, FFT, and run
        the flat MIMO detector on every subcarrier with that subcarrier's channel matrix.

        Y : (Nr, samples) received samples, whole blocks (CP included) per antenna.
        H : time-domain taps (Nr, Nt, L) (needs L - 1 <= cyclic_prefix), or, with freq_domain=True,
            the per-subcarrier responses (Nr, Nt, n_subcarriers) = FFT of the taps (unnormalised).
        method : "zf", "mmse" (unbiased) or "sic" (ordered V-BLAST; OFDM only). N0 is needed for mmse;
                 optional for sic (ZF nulling without it).
        Returns the (Nt, T) detected indices, T = n_blocks * n_subcarriers in transmit order (1-D for
        SISO input); with return_soft=True also the equalized symbols for zf / mmse.
        """
        wf = self.waveform
        if wf.n_subcarriers is None:
            raise ValueError("ofdm_mimo_detect needs an OFDM (or SC with cyclic prefix) waveform.")
        method = str(method).lower()
        if method not in ("zf", "mmse", "sic"):
            raise ValueError("method must be 'zf', 'mmse' or 'sic'.")
        if method == "sic" and wf.name != "OFDM":
            raise ValueError("SIC is only available for OFDM; use zf or mmse for SC-FDE.")
        if method == "mmse" and (N0 is None or N0 < 0):
            raise ValueError("MMSE detection needs N0 >= 0.")
        n = wf.n_subcarriers
        siso = np.ndim(Y) == 1 and np.ndim(H) == 1
        H = np.asarray(H, dtype=complex)
        if H.ndim == 1:
            H = H[None, None, :]
        elif H.ndim == 2:
            H = H[:, :, None]
        Y = np.atleast_2d(np.asarray(Y, dtype=complex))
        if H.ndim != 3 or Y.ndim != 2 or Y.shape[0] != H.shape[0] or Y.size == 0:
            raise ValueError("Y must be (Nr, samples) and H must be (Nr, Nt, L) with the same Nr.")
        if freq_domain:
            if H.shape[2] != n:
                raise ValueError("With freq_domain=True, H must be (Nr, Nt, n_subcarriers).")
            Hf = H
        else:
            if H.shape[2] - 1 > wf.cyclic_prefix:
                raise ValueError("The channel memory L-1 exceeds the cyclic prefix; per-subcarrier "
                                 "equalization would leave ISI.")
            Hf = np.fft.fft(H, n, axis=2)
        Nr, Nt = Hf.shape[:2]
        Yd = np.vstack([wf.demodulate(row) for row in Y])        # CP removed (+ FFT for OFDM)
        if wf.name == "OFDM":
            Yf = Yd
        else:
            Yf = (np.fft.fft(Yd.reshape(Nr, -1, n), axis=2) / np.sqrt(n)).reshape(Nr, -1)
        T = Yf.shape[1]
        plain = Receiver(self.constellation)                      # flat detectors, no demodulation
        soft = np.zeros((Nt, T), dtype=complex)
        idx = np.zeros((Nt, T), dtype=int)
        for k in range(n):
            Yk, Hk = Yf[:, k::n], Hf[:, :, k]
            if method == "zf":
                soft[:, k::n] = plain.zf_mimo(Yk, Hk, power_split)
            elif method == "mmse":
                soft[:, k::n] = plain.mmse_mimo(Yk, Hk, N0, power_split, unbiased=True)
            else:
                idx[:, k::n] = plain.sic_vblast(Yk, Hk, N0, power_split, ordered)[0]
        if method != "sic":
            if wf.name == "SC":                                  # SC-FDE: back to the time domain
                soft = (np.fft.ifft(soft.reshape(Nt, -1, n), axis=2) * np.sqrt(n)).reshape(Nt, -1)
            idx = self._slice(soft)
        if siso:
            soft, idx = soft[0], idx[0]
        return (idx, soft) if return_soft and method != "sic" else idx

    def decode_ofdm_mimo(self, Y, H, **kwargs):
        """Bits from ofdm_mimo_detect (V-BLAST stream order for MIMO)."""
        idx = self.ofdm_mimo_detect(Y, H, **kwargs)
        return self.bit_table[idx if idx.ndim == 1 else self._stream(idx)].ravel()

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


