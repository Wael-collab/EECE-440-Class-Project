"""
EECE 640 class project - INTEGRATION: general end-to-end link engine.

Headless (no Streamlit) glue between the Transmitter, Channel and Receiver modules.
It builds ONE link from a LinkConfig and runs Monte Carlo frames through

    bits -> Constellation -> CodingScheme -> Waveform (per antenna)      [Transmitter/]
         -> channel model + noise                                         [Channel/]
         -> waveform demodulation + detector -> bits                      [Receiver/]

Only code that already exists in the three modules is used; this file only decides,
for every combination, WHICH existing function is called, with which array shapes and
with which convention. The GUI page (pages/1_General_Link_Simulator.py) and the
integration test (integration_test.py) both use it.

Conventions (identical for every combination)
---------------------------------------------
* Y = H X + N, X of shape (Nt, channel uses) (rows = antennas), H of shape (Nr, Nt)
  (flat) or (Nr, Nt, L) (multipath); noise CN(0, N0) per complex receive sample.
* The constellation has average energy A. With power_split=True (default, the
  transmitter convention) the TOTAL energy per channel use is A for every Nt; with
  power_split=False each antenna sends energy A (total Nt*A).
* SNR definitions:  "EsN0": N0 = A / 10^(SNR/10)  (same as the original controller);
                    "EbN0": N0 = Eb / 10^(SNR/10), Eb = transmitted energy per
                    information bit, INCLUDING cyclic-prefix, reference-symbol and
                    repetition overhead and the total over all transmit antennas.
* alpha-stable impulsive noise (Channel/AWSN.py) replaces the Gaussian noise: the
  channel is called with Nzero = 0 and SaS noise of scale gamma is added. N0 is
  undefined for alpha < 2; detectors that need a noise level (MMSE, energy detection)
  are given the alpha = 2 equivalent N0 = 4 gamma^2, and a sweep maps the
  Gaussian-equivalent SNR to gamma = sqrt(N0 / 4).
* Perfect CSI: the receiver is handed the channel realization returned by the
  channel function (pilot-based estimation is not connected on purpose).
"""

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from Transmitter.Transmitter import Constellation, Transmitter, Waveform
from Transmitter.Coding_Schemes import (
    Repetition, Alamouti, VBLAST, DifferentialPSK, DifferentialAlamouti,
    SpaceTimeTransmitter,
)
from Channel.AWGNChannel import AWGN_Channel
from Channel.Rayleigh import Rayleigh_Channel
from Channel.RicianChannel import Rician_Channel
from Channel.MIMORicianFlatChannel import MIMORicianFlatChannel
from Channel.MIMOLTapChannel import MIMOLTapChannel
from Channel.OFDMChannel import OFDM as LegacyOFDMLink
from Receiver.receiver import Receiver, fsk_encode, theory_ser_qam

try:
    from Channel.AIndependentSN import Alpha_Stable_Channel
except ModuleNotFoundError as _exc:          # optional dependency aub_htp
    if _exc.name != "aub_htp":
        raise
    Alpha_Stable_Channel = None

ALPHA_STABLE_AVAILABLE = Alpha_Stable_Channel is not None


# ======================================================================
# Catalogues (key -> label); the GUI shows the labels, the engine uses the keys
# ======================================================================

CONSTELLATIONS = {
    "PAM": "PAM (Gray)",
    "QAM": "Square QAM (Gray)",
    "PSK": "PSK (Gray)",
    "ASK": "Unipolar ASK / OOK (custom mapping, levels 0..M-1)",
    "CUSTOM": "Custom bit-to-symbol mapping",
}

SCHEMES = {
    "SISO": "Single stream (SISO / SIMO), no space-time code",
    "REPETITION": "Repetition code (Coding_Schemes.Repetition)",
    "ALAMOUTI": "Alamouti STBC, 2 Tx (Coding_Schemes.Alamouti)",
    "VBLAST": "V-BLAST spatial multiplexing (Coding_Schemes.VBLAST)",
    "DPSK": "Differential PSK, non-coherent (Coding_Schemes.DifferentialPSK)",
    "DIFF_ALAMOUTI": "Differential Alamouti, non-coherent, 2 Tx",
    "FSK": "Orthogonal M-FSK, non-coherent (receiver.fsk_encode)",
}

WAVEFORMS = {
    "SC": "Single carrier, no cyclic prefix",
    "SC_CP": "Single carrier with cyclic prefix (SC-FDE)",
    "OFDM": "OFDM with cyclic prefix",
}

CHANNELS = {
    "AWGN": "AWGN (fixed unit-gain H, no fading)",
    "RAYLEIGH_FAST": "Rayleigh, fast fading per sample (Rayleigh.py)",
    "RICIAN_FAST": "Rician, fast fading per sample (RicianChannel.py)",
    "RAYLEIGH_BLOCK": "Rayleigh, flat block fading MIMO (MIMORicianFlatChannel, K=0)",
    "RICIAN_BLOCK": "Rician, flat block fading MIMO (MIMORicianFlatChannel)",
    "MULTIPATH": "Multipath L-tap Rayleigh MIMO (MIMOLTapChannel)",
    "LEGACY_OFDM": "Legacy self-contained OFDM link (OFDMChannel.OFDM)",
}

NOISES = {
    "GAUSSIAN": "Gaussian (AWGN)",
    "ALPHA_STABLE": "Symmetric alpha-stable impulsive (AWSN.py)",
}

LOS_MODELS = {
    "DFT": "DFT-phase LoS (full rank, as in the original controller)",
    "ONES": "All-ones LoS (rank one, channel-module default)",
}

DETECTORS = {
    # flat MIMO / SIMO (coherent)
    "ZF": "Zero forcing (decode_mimo zf)",
    "MMSE": "MMSE (decode_mimo mmse)",
    "SIC_ZF": "Ordered SIC / V-BLAST, ZF nulling",
    "SIC_MMSE": "Ordered SIC / V-BLAST, MMSE nulling",
    "ML": "Joint ML, exhaustive (mlse_detect_mimo, L = 1)",
    "MRC": "Maximum ratio combining",
    "EGC": "Equal gain combining",
    "SEL": "Selection combining",
    "SCALAR_ZF": "Scalar zero forcing y/h",
    "SCALAR_MMSE": "Scalar MMSE (unbiased)",
    # single-carrier ISI equalizers
    "LIN_ZF": "Linear ZF equalizer (time domain)",
    "LIN_MMSE": "Linear MMSE equalizer (time domain)",
    "DFE_ZF": "Decision-feedback equalizer, ZF",
    "DFE_MMSE": "Decision-feedback equalizer, MMSE",
    "MLSE_VITERBI": "MLSE, full Viterbi",
    "MLSE_MALG": "MLSE, M-algorithm (reduced state)",
    # frequency-domain (CP) detectors
    "FDE_ZF": "SC-FDE, per-frequency ZF",
    "FDE_MMSE": "SC-FDE, per-frequency MMSE",
    "OFDM_ZF": "OFDM per-subcarrier ZF",
    "OFDM_MMSE": "OFDM per-subcarrier MMSE",
    "OFDM_SIC": "OFDM per-subcarrier ordered SIC (MMSE nulling)",
    "ONE_TAP_ZF": "One-tap ZF per subcarrier",
    "ONE_TAP_MMSE": "One-tap MMSE per subcarrier (unbiased)",
    # space-time / non-coherent
    "ALAMOUTI": "Alamouti combining",
    "DIFF": "Differential detection",
    "DIFF_ALAMOUTI": "Differential Alamouti detection",
    "FSK": "Non-coherent orthogonal FSK (energy per tone)",
    "ENERGY": "Energy (envelope) detection",
}

SNR_DEFINITIONS = {
    "EsN0": "Es/N0 (A / N0, per channel use, as in the original controller)",
    "EbN0": "Eb/N0 (transmitted energy per information bit / N0)",
}

FAST = ("RAYLEIGH_FAST", "RICIAN_FAST")
BLOCK = ("RAYLEIGH_BLOCK", "RICIAN_BLOCK")
FLAT = ("AWGN",) + BLOCK
ML_MAX_JOINT = 1 << 12              # exhaustive joint ML: M^Nt candidates per symbol
TRELLIS_MAX_STATES = 1 << 16        # same limits as Receiver.mlse_detect_mimo
TRELLIS_MAX_BRANCHES = 1 << 22


@dataclass
class LinkConfig:
    """Every user choice of one end-to-end link (see the catalogues above)."""
    constellation: str = "QAM"
    M: int = 4
    A: float = 1.0
    custom_mapping: Optional[dict] = None     # {"00": 1+1j, ...} for CUSTOM
    scheme: str = "SISO"
    n_tx: int = 1                             # used by V-BLAST (others fix it)
    n_rx: int = 1
    rep_L: int = 2                            # Repetition factor
    power_split: bool = True
    waveform: str = "SC"
    n_subcarriers: int = 64                   # OFDM size / SC-CP block size
    cyclic_prefix: int = 16
    channel: str = "AWGN"
    K: float = 1.0                            # Rician K-factor (linear)
    los: str = "DFT"
    L_taps: int = 3                           # MULTIPATH and LEGACY_OFDM
    legacy_block: int = 64                    # LEGACY_OFDM block length
    noise: str = "GAUSSIAN"
    alpha: float = 1.8
    detectors: list = field(default_factory=lambda: ["ZF"])
    # MLSE options (Receiver.mlse_detect_mimo)
    mlse_n_keep: int = 16
    mlse_n_cand: Optional[int] = None
    mlse_shorten: Optional[int] = None


# ======================================================================
# Building blocks
# ======================================================================

def ask_mapping(M):
    """Unipolar ASK (OOK for M = 2): Gray labels on the levels 0, 1, ..., M-1."""
    if M < 2 or (M & (M - 1)) != 0:
        raise ValueError("ASK needs M to be a power of 2.")
    k = int(np.log2(M))
    return {format(m ^ (m >> 1), f"0{k}b"): float(m) for m in range(M)}


def parse_mapping(text):
    """'00: 1+1j, 01: -1+1j, ...' (or one entry per line) -> {label: complex}."""
    mapping = {}
    for part in str(text).replace(";", ",").replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Entry '{part}' must look like 'label: value'.")
        label, value = (s.strip() for s in part.split(":", 1))
        try:
            mapping[label] = complex(value.replace(" ", "").replace("i", "j"))
        except ValueError:
            raise ValueError(f"Could not read the symbol value '{value}'.") from None
    if len(mapping) < 2:
        raise ValueError("Enter at least two 'label: value' entries.")
    return mapping


def make_constellation(cfg):
    """The Constellation object (always normalized to average energy A)."""
    if cfg.scheme == "FSK":                  # FSK only needs M and the Gray bit table
        return Constellation("PAM", M=cfg.M, A=cfg.A)
    if cfg.constellation in ("PAM", "QAM", "PSK"):
        return Constellation(cfg.constellation, M=cfg.M, A=cfg.A)
    if cfg.constellation == "ASK":
        return Constellation("ASK", mapping=ask_mapping(cfg.M), A=cfg.A)
    if cfg.constellation == "CUSTOM":
        if not cfg.custom_mapping:
            raise ValueError("Enter a custom mapping.")
        return Constellation("CUSTOM", mapping=dict(cfg.custom_mapping), A=cfg.A)
    raise ValueError(f"Unknown constellation '{cfg.constellation}'.")


def make_waveform(cfg):
    if cfg.waveform == "SC":
        return Waveform("SC")
    if cfg.waveform == "SC_CP":
        return Waveform("SC", n_subcarriers=int(cfg.n_subcarriers),
                        cyclic_prefix=int(cfg.cyclic_prefix))
    if cfg.waveform == "OFDM":
        return Waveform("OFDM", n_subcarriers=int(cfg.n_subcarriers),
                        cyclic_prefix=int(cfg.cyclic_prefix))
    raise ValueError(f"Unknown waveform '{cfg.waveform}'.")


def make_scheme(cfg):
    """The CodingScheme object (None for FSK, which is a modulation, not a code)."""
    s = cfg.scheme
    if s == "SISO":
        return VBLAST(1)                      # one stream, one antenna, no scaling
    if s == "REPETITION":
        return Repetition(int(cfg.rep_L))
    if s == "ALAMOUTI":
        return Alamouti(power_split=cfg.power_split)
    if s == "VBLAST":
        return VBLAST(int(cfg.n_tx), power_split=cfg.power_split)
    if s == "DPSK":
        return DifferentialPSK()
    if s == "DIFF_ALAMOUTI":
        return DifferentialAlamouti(power_split=cfg.power_split)
    if s == "FSK":
        return None
    raise ValueError(f"Unknown scheme '{s}'.")


def n_tx_of(cfg):
    if cfg.scheme in ("ALAMOUTI", "DIFF_ALAMOUTI"):
        return 2
    if cfg.scheme == "VBLAST":
        return int(cfg.n_tx)
    return 1


def los_matrix(n_rx, n_tx):
    """Deterministic unit-modulus LoS matrix of the original controller (DFT phases)."""
    return np.exp(1j * 2 * np.pi * np.outer(np.arange(n_rx), np.arange(n_tx)) / n_rx)


def constant_modulus(c):
    mags = np.abs(c.table)
    return bool(np.allclose(mags, mags[0]))


def distinct_magnitudes(c):
    return len(np.unique(np.round(np.abs(c.table) ** 2, 12))) == c.M


# ======================================================================
# Which combinations exist (used by the GUI to offer only valid choices)
# ======================================================================

def waveforms_for(scheme):
    if scheme in ("DPSK", "DIFF_ALAMOUTI", "FSK"):
        return ["SC"]
    return list(WAVEFORMS)


def channels_for(scheme, waveform):
    if scheme == "SISO":
        out = ["AWGN", *FAST, *BLOCK, "MULTIPATH"]
        if waveform == "SC":
            out.append("LEGACY_OFDM")
        return out
    if scheme == "REPETITION":
        out = ["AWGN", *BLOCK, "MULTIPATH"]
        if waveform != "OFDM":
            out[1:1] = list(FAST)
        if waveform == "SC":
            out.append("LEGACY_OFDM")
        return out
    if scheme == "ALAMOUTI":
        return ["AWGN", *BLOCK] + (["MULTIPATH"] if waveform == "OFDM" else [])
    if scheme == "VBLAST":
        return ["AWGN", *BLOCK, "MULTIPATH"]
    if scheme == "DPSK":
        return ["AWGN", *FAST, *BLOCK]
    if scheme in ("DIFF_ALAMOUTI", "FSK"):
        return ["AWGN", *BLOCK]
    raise ValueError(f"Unknown scheme '{scheme}'.")


def noises_for(channel):
    if channel == "LEGACY_OFDM":              # it adds its own Gaussian noise inside
        return ["GAUSSIAN"]
    return ["GAUSSIAN", "ALPHA_STABLE"] if ALPHA_STABLE_AVAILABLE else ["GAUSSIAN"]


def viterbi_states(cfg, constellation=None):
    """(states, branches) of the full Viterbi trellis: (M^Nt)^(L_eff - 1), M^Nt."""
    c = constellation if constellation is not None else make_constellation(cfg)
    Mv = c.M ** n_tx_of(cfg)
    Leff = int(cfg.mlse_shorten) if cfg.mlse_shorten else int(cfg.L_taps)
    return Mv ** (Leff - 1), Mv


def viterbi_fits(cfg, constellation=None):
    states, Mv = viterbi_states(cfg, constellation)
    return states <= TRELLIS_MAX_STATES and states * Mv * int(cfg.n_rx) <= TRELLIS_MAX_BRANCHES


def detectors_for(cfg, constellation=None):
    """Detector keys that are valid for this configuration (antennas included)."""
    c = constellation if constellation is not None else make_constellation(cfg)
    s, ch, wf = cfg.scheme, cfg.channel, cfg.waveform
    nt, nr = n_tx_of(cfg), int(cfg.n_rx)
    zf_ok = nr >= nt
    out = []
    if s in ("SISO", "VBLAST"):
        if ch in FLAT:
            out += ["ZF"] if zf_ok else []
            out += ["MMSE"]
            out += ["SIC_ZF"] if zf_ok else []
            out += ["SIC_MMSE"]
            if wf == "SC" and c.M ** nt <= ML_MAX_JOINT:
                out += ["ML"]
            if nt == 1:
                out += ["MRC", "EGC", "SEL"]
                if distinct_magnitudes(c) and wf != "OFDM":
                    out += ["ENERGY"]
        elif ch in FAST:                      # SISO only (Nt = 1)
            out += ["MRC", "EGC", "SEL"]
            if nr == 1:
                out += ["SCALAR_ZF", "SCALAR_MMSE"]
            if distinct_magnitudes(c) and wf != "OFDM":
                out += ["ENERGY"]
        elif ch == "MULTIPATH":
            if wf == "SC":
                out += (["LIN_ZF"] if zf_ok else []) + ["LIN_MMSE"]
                out += (["DFE_ZF"] if zf_ok else []) + ["DFE_MMSE"]
                if viterbi_fits(cfg, c):
                    out += ["MLSE_VITERBI"]
                out += ["MLSE_MALG"]
            elif wf == "SC_CP":
                out += (["FDE_ZF"] if zf_ok else []) + ["FDE_MMSE"]
            else:
                out += (["OFDM_ZF"] if zf_ok else []) + ["OFDM_MMSE", "OFDM_SIC"]
        elif ch == "LEGACY_OFDM":
            out += ["ONE_TAP_ZF", "ONE_TAP_MMSE"] if nr == 1 else []
            out += ["MRC", "EGC", "SEL"]
    elif s == "REPETITION":
        if ch in FLAT or ch in FAST or ch == "LEGACY_OFDM":
            out += ["MRC", "EGC", "SEL"]
        elif ch == "MULTIPATH":
            if wf == "OFDM":
                out += ["MRC", "EGC", "SEL"]
            elif wf == "SC_CP":
                out += ["FDE_ZF", "FDE_MMSE"]
            else:
                out += ["LIN_ZF", "LIN_MMSE", "DFE_ZF", "DFE_MMSE"]
    elif s == "ALAMOUTI":
        out += ["ALAMOUTI"]
    elif s == "DPSK":
        out += ["DIFF"]
    elif s == "DIFF_ALAMOUTI":
        out += ["DIFF_ALAMOUTI"]
    elif s == "FSK":
        out += ["FSK"]
    return out


def symbol_multiple(cfg):
    """The number of information symbols per frame must be a multiple of this."""
    n = int(cfg.n_subcarriers) if cfg.waveform in ("SC_CP", "OFDM") else 1
    s = cfg.scheme
    if cfg.channel == "LEGACY_OFDM":
        n = int(cfg.legacy_block)
    if s == "SISO":
        return n
    if s == "REPETITION":
        L = int(cfg.rep_L)
        return n // math.gcd(n, L)
    if s == "ALAMOUTI":
        return 2 * n if cfg.waveform == "OFDM" else (2 * n) // math.gcd(2, n)
    if s == "VBLAST":
        return int(cfg.n_tx) * n
    if s == "DIFF_ALAMOUTI":
        return 2
    return 1                                   # DPSK, FSK


def bit_multiple(cfg):
    return int(np.log2(cfg.M)) * symbol_multiple(cfg)


def validate(cfg):
    """Raise ValueError with a readable message if the combination is not supported."""
    for key, table in (("scheme", SCHEMES), ("waveform", WAVEFORMS),
                       ("channel", CHANNELS), ("noise", NOISES)):
        if getattr(cfg, key) not in table:
            raise ValueError(f"Unknown {key} '{getattr(cfg, key)}'.")
    c = make_constellation(cfg)
    if cfg.waveform not in waveforms_for(cfg.scheme):
        raise ValueError(f"{SCHEMES[cfg.scheme]} only works with: "
                         + ", ".join(WAVEFORMS[w] for w in waveforms_for(cfg.scheme)) + ".")
    if cfg.channel not in channels_for(cfg.scheme, cfg.waveform):
        raise ValueError(f"{CHANNELS[cfg.channel]} is not available for "
                         f"{SCHEMES[cfg.scheme]} with {WAVEFORMS[cfg.waveform]}.")
    if cfg.noise not in noises_for(cfg.channel):
        raise ValueError("alpha-stable noise needs the aub_htp package and is not "
                         "available for the legacy OFDM link.")
    if cfg.scheme in ("DPSK", "DIFF_ALAMOUTI") and not constant_modulus(c):
        raise ValueError("Differential schemes need a constant-modulus constellation (PSK).")
    if cfg.scheme == "VBLAST" and int(cfg.n_tx) < 2:
        raise ValueError("V-BLAST needs Nt >= 2 (use the single-stream scheme for Nt = 1).")
    if int(cfg.n_rx) < 1:
        raise ValueError("Nr must be >= 1.")
    if cfg.scheme == "REPETITION" and int(cfg.rep_L) < 1:
        raise ValueError("The repetition factor L must be >= 1.")
    if cfg.channel in ("MULTIPATH", "LEGACY_OFDM") and int(cfg.L_taps) < 1:
        raise ValueError("The number of taps L must be >= 1.")
    if (cfg.channel == "MULTIPATH" and cfg.waveform in ("SC_CP", "OFDM")
            and int(cfg.L_taps) - 1 > int(cfg.cyclic_prefix)):
        raise ValueError("The cyclic prefix must be at least L - 1 for the multipath channel.")
    if cfg.noise == "ALPHA_STABLE" and not 0 < float(cfg.alpha) <= 2:
        raise ValueError("alpha must be in (0, 2].")
    allowed = detectors_for(cfg, c)
    if not cfg.detectors:
        raise ValueError("Select at least one detector.")
    bad = [d for d in cfg.detectors if d not in allowed]
    if bad:
        raise ValueError("Not available for this configuration: "
                         + ", ".join(DETECTORS.get(d, d) for d in bad) + ".")
    if cfg.mlse_shorten is not None and not 1 <= int(cfg.mlse_shorten) <= int(cfg.L_taps):
        raise ValueError("'shorten' must be between 1 and L.")
    if cfg.mlse_n_cand is not None and int(cfg.n_rx) < n_tx_of(cfg):
        raise ValueError("K-best sphere candidates (n_cand) need Nr >= Nt.")
    return True


# ======================================================================
# SNR / energy bookkeeping
# ======================================================================

def energy_per_channel_use(cfg):
    """Nominal TOTAL transmit energy per channel use (sum over antennas)."""
    nt = n_tx_of(cfg)
    if nt > 1 and not cfg.power_split:
        return nt * float(cfg.A)
    return float(cfg.A)


def frame_dimensions(cfg, n_bits):
    """(channel uses per antenna incl. CP / reference symbols, transmitted energy)."""
    k = int(np.log2(cfg.M))
    Ns = n_bits // k
    if cfg.scheme == "FSK":
        return cfg.M * Ns, float(cfg.A) * Ns          # M orthogonal dimensions per symbol
    T = make_scheme(cfg).channel_uses(Ns)
    if cfg.waveform in ("SC_CP", "OFDM"):
        n, cp = int(cfg.n_subcarriers), int(cfg.cyclic_prefix)
        T = T // n * (n + cp)
    elif cfg.channel == "LEGACY_OFDM":
        n, cp = int(cfg.legacy_block), int(cfg.L_taps)   # it inserts a CP of length L
        T = T // n * (n + cp)
    return T, energy_per_channel_use(cfg) * T


def n0_from_snr(cfg, snr_db, n_bits, definition="EsN0"):
    snr = 10.0 ** (float(snr_db) / 10.0)
    if definition == "EsN0":
        return float(cfg.A) / snr
    if definition == "EbN0":
        _, energy = frame_dimensions(cfg, n_bits)
        return energy / n_bits / snr
    raise ValueError("definition must be 'EsN0' or 'EbN0'.")


def gamma_from_n0(N0):
    """alpha = 2 equivalence used for sweeps with alpha-stable noise: N0 = 4 gamma^2."""
    return math.sqrt(max(N0, 0.0) / 4.0)


# ======================================================================
# The link
# ======================================================================

class Link:
    """One configured transmitter-channel-receiver chain."""

    def __init__(self, cfg):
        validate(cfg)
        self.cfg = cfg
        self.c = make_constellation(cfg)
        self.k = self.c.bits_per_symbol
        self.waveform = make_waveform(cfg)
        self.tx = Transmitter(self.c, self.waveform)
        self.rx = Receiver(self.c, self.waveform)
        self.plain = Receiver(self.c)                # symbol domain, no demodulation
        self.scheme = make_scheme(cfg)
        self.st = None if self.scheme is None else SpaceTimeTransmitter(self.tx, self.scheme)
        self.n_tx = n_tx_of(cfg)
        self.n_rx = int(cfg.n_rx)
        self.ps = bool(cfg.power_split)
        self.H_aw = los_matrix(self.n_rx, self.n_tx)   # AWGN: fixed unit-modulus H

    # ---------------- transmitter ----------------
    def transmit(self, bits):
        N = len(bits)
        if self.cfg.scheme == "FSK":
            self.tx.symbols(bits, N)                         # same input checks as run()
            idx = self.rx.bits_to_indices(bits)
            return fsk_encode(idx, self.c.M, self.c.A)      # (M, Ns) correlator domain
        return self.st.run(bits, N)                          # (Nt, samples)

    # ---------------- channel + noise ----------------
    def _noise(self, clean, N0, gamma):
        if self.cfg.noise == "GAUSSIAN":
            return np.asarray(AWGN_Channel(np.asarray(clean, dtype=complex), N0))
        flat = Alpha_Stable_Channel(np.asarray(clean, dtype=complex).ravel(),
                                    float(self.cfg.alpha), float(gamma))
        return np.asarray(flat).reshape(np.shape(clean))

    def propagate(self, X, N0, gamma=0.0):
        """Return (Y, H). Shapes depend on the channel (see detect())."""
        cfg, nr, nt = self.cfg, self.n_rx, self.n_tx
        gauss = cfg.noise == "GAUSSIAN"
        n0_in = N0 if gauss else 0.0
        fsk = cfg.scheme == "FSK"
        Xs = X.reshape(1, -1) if fsk else X         # FSK: one "antenna", M*Ns dimensions

        if cfg.channel == "AWGN":
            H = np.ones((nr, 1), dtype=complex) if fsk else self.H_aw
            Y = self._noise(H @ Xs, N0, gamma)
        elif cfg.channel in FAST:
            rows, gains = [], []
            for _ in range(nr):
                if cfg.channel == "RAYLEIGH_FAST":
                    y, h = Rayleigh_Channel(Xs[0], n0_in)
                else:
                    y, h = Rician_Channel(Xs[0], n0_in, K=float(cfg.K))
                rows.append(y)
                gains.append(h)
            Y, H = np.vstack(rows), np.vstack(gains)
            if not gauss:
                Y = self._noise(Y, N0, gamma)
        elif cfg.channel in BLOCK:
            K = 0.0 if cfg.channel == "RAYLEIGH_BLOCK" else float(cfg.K)
            ntx = 1 if fsk else nt
            H_los = los_matrix(nr, ntx) if cfg.los == "DFT" else None
            Y, H = MIMORicianFlatChannel(Xs, n0_in, ntx, nr, K=K, H_los=H_los)
            if not gauss:
                Y = self._noise(Y, N0, gamma)
        elif cfg.channel == "MULTIPATH":
            Y, H = MIMOLTapChannel(int(cfg.L_taps), Xs, n0_in, nt, nr)
            if not gauss:
                Y = self._noise(Y, N0, gamma)
        elif cfg.channel == "LEGACY_OFDM":
            # Self-contained OFDM link per receive antenna (independent taps per antenna);
            # its output is already in the frequency domain, with gains FFT(h).
            n = int(cfg.legacy_block)
            Y = np.zeros((nr, Xs.shape[1]), dtype=complex)
            H = np.zeros_like(Y)
            for r in range(nr):
                for b in range(Xs.shape[1] // n):
                    blk = Xs[0, b * n:(b + 1) * n]
                    y, h = LegacyOFDMLink(blk, int(cfg.L_taps), N0)
                    Y[r, b * n:(b + 1) * n] = y
                    H[r, b * n:(b + 1) * n] = np.fft.fft(h, n)
        else:
            raise ValueError(f"Unknown channel '{cfg.channel}'.")
        if fsk:
            Y = Y.reshape(nr, self.c.M, -1)
        return np.asarray(Y), np.asarray(H)

    # ---------------- receiver ----------------
    def _symbol_gains(self, H):
        """Per-sample gains (Nr, samples) -> symbol-domain gains for SC (CP removed)."""
        return np.vstack([self.waveform.demodulate(row) for row in H])

    def _combine(self, key, Y, H, N0):
        """Coherent SIMO combining on (Nr, T) data with gains H (Nr,), (Nr,1) or (Nr,T)."""
        r = self.plain
        if key == "MRC":
            return r.mrc_combine(Y, H)
        if key == "EGC":
            return r.egc_combine(Y, H)
        if key == "SEL":
            return r.selection_combine(Y, H)
        if key == "SCALAR_ZF":
            return r.zf_equalize(Y[0], np.asarray(H)[0])
        if key == "SCALAR_MMSE":
            return r.mmse_equalize(Y[0], np.asarray(H)[0], N0, unbiased=True)
        raise ValueError(key)

    def _bits(self, soft):
        return self.rx.bit_table[self.rx._slice(soft)].ravel()

    def detect(self, key, Y, H, N0):
        """Bits (in the transmitted order) from one detector."""
        cfg, rx, wf = self.cfg, self.rx, self.waveform
        s, ch = cfg.scheme, cfg.channel
        ps = self.ps
        S = None
        if ch == "MULTIPATH" and cfg.waveform in ("SC_CP", "OFDM"):
            L = H.shape[2]
            Y = Y[:, :Y.shape[1] - (L - 1)]          # drop the convolution tail (CP absorbs ISI)

        if s in ("SISO", "VBLAST"):
            if ch in FLAT:
                if key in ("ZF", "MMSE"):
                    return rx.decode_mimo(Y, H, key.lower(), N0, ps)
                if key == "SIC_ZF":
                    return rx.decode_mimo(Y, H, "sic", None, ps)
                if key == "SIC_MMSE":
                    return rx.decode_mimo(Y, H, "sic", N0, ps)
                if key == "ML":
                    return rx.decode_mlse_mimo(Y, H, tail=False, power_split=ps)
                if key in ("MRC", "EGC", "SEL"):
                    return getattr(rx, {"MRC": "decode_mrc", "EGC": "decode_egc",
                                        "SEL": "decode_selection"}[key])(Y, H[:, 0])
                if key == "ENERGY":
                    if ch == "AWGN":
                        return rx.decode_energy(Y, N0, None)
                    # fading ML energy detector needs N0 > 0 (E|h|^2 = 1 for every model)
                    return rx.decode_energy(Y, max(N0, 1e-12 * cfg.A), 1.0)
            elif ch in FAST:
                if key == "ENERGY":
                    return rx.decode_energy(Y, max(N0, 1e-12 * cfg.A), 1.0)
                if cfg.waveform == "OFDM":
                    # per-sample gains do not give one gain per subcarrier: equalize /
                    # combine in the time domain first, then remove the CP and FFT
                    # (the "sample-domain coherent" approach of flow_test.py).
                    z = self._combine(key, Y, H, N0)
                    return self._bits(wf.demodulate(z))
                Hs = self._symbol_gains(H)
                Yd = np.vstack([wf.demodulate(row) for row in Y])
                return self._bits(self._combine(key, Yd, Hs, N0))
            elif ch == "MULTIPATH":
                if key in ("LIN_ZF", "LIN_MMSE", "DFE_ZF", "DFE_MMSE"):
                    kind, structure = key.split("_")[1].lower(), (
                        "linear" if key.startswith("LIN") else "dfe")
                    return rx.decode_sc(Y, H, N0=N0, tail=True, kind=kind,
                                        structure=structure, power_split=ps)
                if key in ("MLSE_VITERBI", "MLSE_MALG"):
                    kw = dict(tail=True, power_split=ps,
                              method="viterbi" if key == "MLSE_VITERBI" else "malg")
                    if key == "MLSE_MALG":
                        kw["n_keep"] = int(cfg.mlse_n_keep)
                        if cfg.mlse_n_cand:
                            kw["n_cand"] = int(cfg.mlse_n_cand)
                    if cfg.mlse_shorten:
                        kw["shorten"], kw["N0"] = int(cfg.mlse_shorten), max(N0, 1e-12)
                    return rx.decode_mlse_mimo(Y, H, **kw)
                if key in ("FDE_ZF", "FDE_MMSE", "OFDM_ZF", "OFDM_MMSE", "OFDM_SIC"):
                    method = key.split("_")[1].lower()
                    return rx.decode_ofdm_mimo(Y, H, N0=N0, method=method, power_split=ps)
            elif ch == "LEGACY_OFDM":
                if key == "ONE_TAP_ZF":
                    return rx.decode(Y[0], H[0])
                if key == "ONE_TAP_MMSE":
                    return rx.decode(Y[0], H[0], N0)
                return self._bits(self._combine(key, Y, H, N0))

        elif s == "REPETITION":
            L = int(cfg.rep_L)
            method = {"MRC": "mrc", "EGC": "egc", "SEL": "selection"}.get(key)
            if ch in FLAT:
                return rx.decode_repetition(Y, H[:, 0], L, method)
            if ch in FAST:
                return rx.decode_repetition(Y, self._symbol_gains(H), L, method)
            if ch == "LEGACY_OFDM":
                return rx.decode_repetition(Y, H, L, method)
            if ch == "MULTIPATH":
                if cfg.waveform == "OFDM":
                    n = wf.n_subcarriers
                    Hf = np.fft.fft(H[:, 0, :], n, axis=1)            # (Nr, n)
                    T = Y.shape[1] // (n + wf.cyclic_prefix) * n
                    Hseq = np.tile(Hf, (1, T // n))                     # gain of each symbol
                    return rx.decode_repetition(Y, Hseq, L, method)
                # SC / SC-FDE: equalize every copy, then average the L copies
                if cfg.waveform == "SC_CP":
                    _, soft = rx.ofdm_mimo_detect(Y, H, N0=N0, method=key.split("_")[1].lower(),
                                                  power_split=ps, return_soft=True)
                else:
                    kind = key.split("_")[1].lower()
                    structure = "linear" if key.startswith("LIN") else "dfe"
                    soft = rx.equalize_sc(Y, H, N0=N0, tail=True, kind=kind,
                                          structure=structure, power_split=ps)
                soft = np.asarray(soft).reshape(-1)
                return self._bits(soft.reshape(-1, L).mean(axis=1))

        elif s == "ALAMOUTI":
            if cfg.waveform == "OFDM":
                return rx.decode_alamouti_ofdm(Y, H, power_split=ps)
            return rx.decode_alamouti(Y, H, None, ps)
        elif s == "DPSK":
            return rx.decode_differential(Y)
        elif s == "DIFF_ALAMOUTI":
            return rx.decode_differential_alamouti(Y)
        elif s == "FSK":
            return rx.decode_fsk(Y)
        raise ValueError(f"Detector '{key}' is not wired for this configuration.")

    # ---------------- Monte Carlo ----------------
    def simulate(self, n_bits, N0, n_frames=1, gamma=None, seed=None):
        """
        Run n_frames independent frames (new bits, channel and noise each frame); every
        selected detector processes the same frames. N0 is the Gaussian noise variance
        (for alpha-stable noise give gamma; N0 is then only the detector noise level).
        """
        n_bits, n_frames = int(n_bits), int(n_frames)
        if n_bits <= 0 or n_bits % bit_multiple(self.cfg):
            raise ValueError(f"The frame length must be a positive multiple of "
                             f"{bit_multiple(self.cfg)} bits for this configuration.")
        if self.cfg.noise == "ALPHA_STABLE":
            gamma = gamma_from_n0(N0) if gamma is None else float(gamma)
            N0_det = 4.0 * gamma ** 2
        else:
            gamma, N0_det = 0.0, float(N0)
        if N0 < 0:
            raise ValueError("N0 must be non-negative.")
        if seed is not None:
            np.random.seed(int(seed))   # the channel modules use NumPy's global generator

        dets = list(self.cfg.detectors)
        errors = {d: 0 for d in dets}
        sym_errors = {d: 0 for d in dets}
        last = {}
        for _ in range(n_frames):
            bits = np.random.randint(0, 2, n_bits)
            X = self.transmit(bits)
            Y, H = self.propagate(X, float(N0), gamma)
            tx_idx = self.rx.bits_to_indices(bits)
            decoded = {}
            for d in dets:
                out = np.asarray(self.detect(d, Y, H, N0_det), dtype=int).ravel()
                if out.size != n_bits:
                    raise ValueError(f"{DETECTORS[d]} returned {out.size} bits, expected {n_bits}.")
                errors[d] += int(np.count_nonzero(out != bits))
                sym_errors[d] += int(np.count_nonzero(self.rx.bits_to_indices(out) != tx_idx))
                decoded[d] = out
            last = {"bits": bits, "X": X, "Y": Y, "H": H, "decoded": decoded}

        total_bits = n_bits * n_frames
        total_syms = total_bits // self.k
        uses, energy = frame_dimensions(self.cfg, n_bits)
        rate = n_bits / uses
        res = {}
        for d in dets:
            ber = errors[d] / total_bits
            res[d] = {
                "ber": ber, "ser": sym_errors[d] / total_syms,
                "bit_errors": errors[d], "symbol_errors": sym_errors[d],
                "bits": total_bits, "symbols": total_syms,
                "goodput": rate * (1.0 - ber),
            }
        return {
            "per_detector": res, "last": last, "N0": float(N0), "gamma": gamma,
            "channel_uses_per_frame": uses, "rate": rate,
            "EsN0_dB": 10 * math.log10(self.cfg.A / N0) if N0 > 0 else math.inf,
            "EbN0_dB": 10 * math.log10(energy / n_bits / N0) if N0 > 0 else math.inf,
        }

    def sweep(self, n_bits, snr_db_list, n_frames=1, definition="EsN0", seed=None,
              progress=None):
        """BER / SER versus SNR (Es/N0 or Eb/N0); alpha-stable uses gamma = sqrt(N0/4)."""
        if seed is not None:
            np.random.seed(int(seed))
        out = {"snr_db": [], "N0": [], "per_detector": {d: {"ber": [], "ser": []}
                                                        for d in self.cfg.detectors}}
        for i, snr in enumerate(snr_db_list):
            N0 = n0_from_snr(self.cfg, snr, n_bits, definition)
            r = self.simulate(n_bits, N0, n_frames)
            out["snr_db"].append(float(snr))
            out["N0"].append(N0)
            for d, v in r["per_detector"].items():
                out["per_detector"][d]["ber"].append(v["ber"])
                out["per_detector"][d]["ser"].append(v["ser"])
            if progress is not None:
                progress((i + 1) / len(snr_db_list), snr)
        out["theory_ser"] = self.theory_ser(out["N0"])
        return out

    # ---------------- theory (receiver.theory_ser_qam) ----------------
    def theory_available(self):
        cfg = self.cfg
        return (cfg.constellation == "QAM" and cfg.scheme == "SISO" and self.n_rx == 1
                and cfg.channel == "AWGN" and cfg.noise == "GAUSSIAN" and cfg.M >= 4)

    def theory_ser(self, N0_list):
        """Exact square-QAM SER over AWGN at Es/N0 = A/N0 per data symbol, or None."""
        if not self.theory_available():
            return None
        return [theory_ser_qam(self.cfg.M, self.cfg.A / n0) if n0 > 0 else 0.0
                for n0 in N0_list]


# ======================================================================
# Enumeration of every valid combination (used by integration_test.py and the
# GUI self-test)
# ======================================================================

EXACT_NOISELESS = {
    # detectors that must give zero errors without noise on every channel they allow
    "ZF", "MMSE", "SIC_ZF", "SIC_MMSE", "ML", "MRC", "EGC", "SEL", "SCALAR_ZF",
    "SCALAR_MMSE", "MLSE_VITERBI", "FDE_ZF", "FDE_MMSE", "OFDM_ZF", "OFDM_MMSE",
    "OFDM_SIC", "ONE_TAP_ZF", "ONE_TAP_MMSE", "ALAMOUTI", "DIFF_ALAMOUTI", "FSK",
}


def all_configurations(small=True):
    """Yield (description, LinkConfig) for every scheme x waveform x channel x antenna
    setting the engine supports, with all detectors allowed for it."""
    n, cp = (16, 4) if small else (64, 16)
    antennas = {"SISO": [(1, 1), (1, 2)], "REPETITION": [(1, 1), (1, 2)],
                "ALAMOUTI": [(2, 1), (2, 2)], "VBLAST": [(2, 2), (2, 3)],
                "DPSK": [(1, 1), (1, 2)], "DIFF_ALAMOUTI": [(2, 1), (2, 2)],
                "FSK": [(1, 1), (1, 2)]}
    for scheme in SCHEMES:
        const = "PSK" if scheme in ("DPSK", "DIFF_ALAMOUTI") else "QAM"
        for wf in waveforms_for(scheme):
            for ch in channels_for(scheme, wf):
                for nt, nr in antennas[scheme]:
                    cfg = LinkConfig(constellation=const, M=4, A=1.0, scheme=scheme,
                                     n_tx=nt, n_rx=nr, rep_L=2, waveform=wf,
                                     n_subcarriers=n, cyclic_prefix=cp, channel=ch,
                                     K=2.0, L_taps=3, legacy_block=n, detectors=[])
                    cfg.detectors = detectors_for(cfg)
                    yield (f"{scheme} | {wf} | {ch} | Nt={nt} Nr={nr}", cfg)

    # Variants: amplitude constellations (energy detection), multi-level QAM,
    # power_split=False, more antennas, other repetition factors, MLSE options.
    base = dict(A=2.0, n_subcarriers=n, cyclic_prefix=cp, K=2.0, L_taps=3, legacy_block=n)
    extras = []
    for ch in ("AWGN", "RAYLEIGH_FAST", "RAYLEIGH_BLOCK"):
        extras.append(LinkConfig(constellation="ASK", M=4, scheme="SISO", channel=ch,
                                 n_rx=2, **base))
        extras.append(LinkConfig(constellation="ASK", M=2, scheme="SISO", channel=ch,
                                 n_rx=1, **base))
    for wf in ("SC", "SC_CP", "OFDM"):
        for ch in ("AWGN", "RICIAN_BLOCK", "MULTIPATH"):
            extras.append(LinkConfig(constellation="QAM", M=16, scheme="VBLAST", n_tx=3,
                                     n_rx=3, waveform=wf, channel=ch, power_split=False,
                                     **base))
            extras.append(LinkConfig(constellation="QAM", M=16, scheme="SISO", n_rx=2,
                                     waveform=wf, channel=ch, **base))
            extras.append(LinkConfig(constellation="PSK", M=8, scheme="REPETITION",
                                     rep_L=3, n_rx=2, waveform=wf, channel=ch, **base))
        for ch in ("AWGN", "RICIAN_BLOCK") + (("MULTIPATH",) if wf == "OFDM" else ()):
            extras.append(LinkConfig(constellation="QAM", M=16, scheme="ALAMOUTI", n_rx=2,
                                     waveform=wf, channel=ch, power_split=False, **base))
    extras.append(LinkConfig(constellation="PSK", M=8, scheme="DIFF_ALAMOUTI", n_rx=2,
                             channel="RICIAN_BLOCK", power_split=False, **base))
    extras.append(LinkConfig(constellation="PAM", M=8, scheme="FSK", n_rx=3,
                             channel="RICIAN_BLOCK", **base))
    extras.append(LinkConfig(constellation="PSK", M=4, scheme="SISO", n_rx=2,
                             channel="LEGACY_OFDM", **base))
    extras.append(LinkConfig(constellation="QAM", M=4, scheme="REPETITION", rep_L=2,
                             n_rx=1, channel="LEGACY_OFDM", **base))
    mlse = dict(constellation="QAM", M=4, scheme="VBLAST", n_tx=2, n_rx=2,
                channel="MULTIPATH", **base)
    extras.append(LinkConfig(**mlse, detectors=["MLSE_VITERBI", "MLSE_MALG"],
                             mlse_shorten=2))
    extras.append(LinkConfig(**mlse, detectors=["MLSE_MALG"], mlse_n_cand=4, mlse_n_keep=8))
    for cfg in extras:
        if not cfg.detectors or cfg.detectors == ["ZF"]:
            cfg.detectors = detectors_for(cfg)
        yield (f"{cfg.scheme} | {cfg.waveform} | {cfg.channel} | Nt={n_tx_of(cfg)} "
               f"Nr={cfg.n_rx} | {cfg.M}-{cfg.constellation} ps={cfg.power_split}"
               + (f" L={cfg.rep_L}" if cfg.scheme == "REPETITION" else "")
               + (f" shorten={cfg.mlse_shorten}" if cfg.mlse_shorten else "")
               + (f" n_cand={cfg.mlse_n_cand}" if cfg.mlse_n_cand else ""), cfg)


def self_test(n_frames=2, small=True, seed=11, progress=None):
    """
    Noiseless run of every combination. Returns a list of dict rows:
    {combo, detector, ber, status} with status PASS (exact detector, zero errors),
    FAIL (exact detector with errors, or an exception) or INFO (approximate detector,
    e.g. finite-length linear/DFE equalizers, M-algorithm, or non-coherent detection on
    a fading channel, where some errors without noise are expected).
    """
    rows = []
    combos = list(all_configurations(small))
    for i, (name, cfg) in enumerate(combos):
        try:
            link = Link(cfg)
            nb = bit_multiple(cfg) * max(1, 64 // bit_multiple(cfg))
            res = link.simulate(nb, 0.0, n_frames, seed=seed + i)
            for d, v in res["per_detector"].items():
                exact = d in EXACT_NOISELESS
                if d == "DIFF":                  # differential: exact unless h changes per sample
                    exact = cfg.channel not in FAST
                if d == "ENERGY":                # |h|^2 random under fading -> errors expected
                    exact = cfg.channel == "AWGN"
                status = ("PASS" if v["ber"] == 0 else "FAIL") if exact else "INFO"
                rows.append({"combo": name, "detector": d, "ber": v["ber"], "status": status})
        except Exception as exc:                       # report, do not stop the sweep
            rows.append({"combo": name, "detector": "-", "ber": float("nan"),
                         "status": f"FAIL ({type(exc).__name__}: {exc})"})
        if progress is not None:
            progress((i + 1) / len(combos), name)
    return rows


# ======================================================================
# Human-readable signal chain: which existing function each block calls
# ======================================================================

def _receiver_call(cfg, key):
    s, ch, wf = cfg.scheme, cfg.channel, cfg.waveform
    ps = f"power_split={cfg.power_split}"
    if s in ("SISO", "VBLAST"):
        if ch in FLAT:
            return {
                "ZF": f"Receiver.decode_mimo(Y, H, 'zf', {ps})",
                "MMSE": f"Receiver.decode_mimo(Y, H, 'mmse', N0, {ps})",
                "SIC_ZF": f"Receiver.decode_mimo(Y, H, 'sic', N0=None, {ps})",
                "SIC_MMSE": f"Receiver.decode_mimo(Y, H, 'sic', N0, {ps})",
                "ML": f"Receiver.decode_mlse_mimo(Y, H, tail=False, {ps})  (L = 1: joint ML)",
                "MRC": "Receiver.decode_mrc(Y, h)", "EGC": "Receiver.decode_egc(Y, h)",
                "SEL": "Receiver.decode_selection(Y, h)",
                "ENERGY": "Receiver.decode_energy(Y, N0, fading_var="
                          + ("None)" if ch == "AWGN" else "1.0)"),
            }[key]
        if ch in FAST:
            if key == "ENERGY":
                return "Receiver.decode_energy(Y, N0, fading_var=1.0)"
            name = {"MRC": "mrc_combine", "EGC": "egc_combine", "SEL": "selection_combine",
                    "SCALAR_ZF": "zf_equalize", "SCALAR_MMSE": "mmse_equalize(unbiased)"}[key]
            if wf == "OFDM":
                return (f"Receiver.{name}(Y, h) per SAMPLE -> Waveform.demodulate "
                        "(CP removal + FFT) -> slice")
            return f"Waveform.demodulate rows of Y and h -> Receiver.{name} -> slice"
        if ch == "MULTIPATH":
            if key.startswith(("LIN", "DFE")):
                kind = key.split("_")[1].lower()
                st = "linear" if key.startswith("LIN") else "dfe"
                return (f"Receiver.decode_sc(Y, H, N0, tail=True, kind='{kind}', "
                        f"structure='{st}', {ps})")
            if key.startswith("MLSE"):
                m = "viterbi" if key == "MLSE_VITERBI" else "malg"
                extra = (f", n_keep={cfg.mlse_n_keep}" if m == "malg" else "") + (
                    f", n_cand={cfg.mlse_n_cand}" if (m == "malg" and cfg.mlse_n_cand) else "") + (
                    f", shorten={cfg.mlse_shorten}, N0" if cfg.mlse_shorten else "")
                return f"Receiver.decode_mlse_mimo(Y, H, tail=True, method='{m}'{extra}, {ps})"
            method = key.split("_")[1].lower()
            return (f"drop L-1 tail -> Receiver.decode_ofdm_mimo(Y, H, N0, method='{method}', "
                    f"{ps})")
        if ch == "LEGACY_OFDM":
            return {"ONE_TAP_ZF": "Receiver.decode(y, FFT(h))",
                    "ONE_TAP_MMSE": "Receiver.decode(y, FFT(h), N0)",
                    "MRC": "Receiver.mrc_combine(Y, FFT(h)) -> slice",
                    "EGC": "Receiver.egc_combine(Y, FFT(h)) -> slice",
                    "SEL": "Receiver.selection_combine(Y, FFT(h)) -> slice"}[key]
    if s == "REPETITION":
        m = {"MRC": "mrc", "EGC": "egc", "SEL": "selection"}.get(key)
        if ch in FLAT:
            return f"Receiver.decode_repetition(Y, h, L={cfg.rep_L}, method='{m}')"
        if ch in FAST:
            return (f"CP removal on h -> Receiver.decode_repetition(Y, h, L={cfg.rep_L}, "
                    f"method='{m}')")
        if ch == "LEGACY_OFDM":
            return f"Receiver.decode_repetition(Y, FFT(h), L={cfg.rep_L}, method='{m}')"
        if wf == "OFDM":
            return (f"drop tail -> Receiver.decode_repetition(Y, per-subcarrier FFT(h), "
                    f"L={cfg.rep_L}, method='{m}')")
        if wf == "SC_CP":
            return (f"drop tail -> Receiver.ofdm_mimo_detect(..., method="
                    f"'{key.split('_')[1].lower()}', return_soft=True) -> average the "
                    f"{cfg.rep_L} copies -> slice")
        kind = key.split("_")[1].lower()
        st = "linear" if key.startswith("LIN") else "dfe"
        return (f"Receiver.equalize_sc(Y, H, N0, tail=True, kind='{kind}', structure='{st}') "
                f"-> average the {cfg.rep_L} copies -> slice")
    if s == "ALAMOUTI":
        if wf == "OFDM":
            return f"Receiver.decode_alamouti_ofdm(Y, H, {ps})  (per subcarrier, 2 blocks)"
        return f"Receiver.decode_alamouti(Y, H, {ps})"
    if s == "DPSK":
        return "Receiver.decode_differential(Y)  (no channel knowledge)"
    if s == "DIFF_ALAMOUTI":
        return "Receiver.decode_differential_alamouti(Y)  (no channel knowledge)"
    if s == "FSK":
        return "Receiver.decode_fsk(Y)  (no channel knowledge)"
    return "?"


def chain_description(cfg):
    """List of (block, call) pairs describing the configured end-to-end chain."""
    nt = n_tx_of(cfg)
    rows = []
    if cfg.scheme == "FSK":
        rows.append(("Transmitter", f"Constellation('PAM', M={cfg.M}, A={cfg.A})  (Gray bit "
                                    "table only)"))
        rows.append(("Transmitter", f"Receiver.bits_to_indices -> receiver.fsk_encode(idx, "
                                    f"M={cfg.M}, A) -> X ({cfg.M} tones, Ns)"))
    else:
        if cfg.constellation in ("PAM", "QAM", "PSK"):
            cons = f"Constellation('{cfg.constellation}', M={cfg.M}, A={cfg.A})"
        elif cfg.constellation == "ASK":
            cons = f"Constellation('ASK', mapping=ask_mapping({cfg.M}), A={cfg.A})"
        else:
            cons = f"Constellation('CUSTOM', mapping=<{len(cfg.custom_mapping or {})} labels>, A={cfg.A})"
        rows.append(("Transmitter", cons))
        rows.append(("Transmitter", "Transmitter.symbols(bits, N)  (checks + Gray lookup, "
                                    "before the waveform)"))
        sch = {"SISO": "VBLAST(1)", "REPETITION": f"Repetition(L={cfg.rep_L})",
               "ALAMOUTI": f"Alamouti(power_split={cfg.power_split})",
               "VBLAST": f"VBLAST(n_tx={nt}, power_split={cfg.power_split})",
               "DPSK": "DifferentialPSK()",
               "DIFF_ALAMOUTI": f"DifferentialAlamouti(power_split={cfg.power_split})"}[cfg.scheme]
        rows.append(("Coding scheme", f"{sch}.encode(symbols) -> X ({nt}, T)"))
        if cfg.scheme == "ALAMOUTI" and cfg.waveform == "OFDM":
            rows.append(("Coding scheme", "alamouti_ofdm_map(X, n): pair m -> subcarrier m%n "
                                          "of OFDM blocks 2(m//n), 2(m//n)+1"))
        w = {"SC": "Waveform('SC')",
             "SC_CP": f"Waveform('SC', n_subcarriers={cfg.n_subcarriers}, "
                      f"cyclic_prefix={cfg.cyclic_prefix})",
             "OFDM": f"Waveform('OFDM', n_subcarriers={cfg.n_subcarriers}, "
                     f"cyclic_prefix={cfg.cyclic_prefix})"}[cfg.waveform]
        rows.append(("Waveform", f"{w}.modulate(row) for each of the {nt} antenna rows "
                                 "(SpaceTimeTransmitter.run)"))
    nr = int(cfg.n_rx)
    ch = {
        "AWGN": f"AWGN_Channel(H X, N0), H = fixed {nr}x{1 if cfg.scheme == 'FSK' else nt} "
                "unit-modulus DFT-phase matrix",
        "RAYLEIGH_FAST": f"Rayleigh_Channel(x, N0) once per receive antenna ({nr})",
        "RICIAN_FAST": f"Rician_Channel(x, N0, K={cfg.K}) once per receive antenna ({nr})",
        "RAYLEIGH_BLOCK": f"MIMORicianFlatChannel(X, N0, Nt, Nr={nr}, K=0)",
        "RICIAN_BLOCK": f"MIMORicianFlatChannel(X, N0, Nt, Nr={nr}, K={cfg.K}, "
                        f"H_los={'DFT phases' if cfg.los == 'DFT' else 'all ones'})",
        "MULTIPATH": f"MIMOLTapChannel(L={cfg.L_taps}, X, N0, Nt, Nr={nr})",
        "LEGACY_OFDM": f"OFDMChannel.OFDM(block of {cfg.legacy_block}, L={cfg.L_taps}, N0) per "
                       f"block and receive antenna (its own IDFT, CP = L, L-tap channel, DFT)",
    }[cfg.channel]
    rows.append(("Channel", ch))
    if cfg.noise == "ALPHA_STABLE":
        rows.append(("Channel", f"channel with Nzero = 0, then Alpha_Stable_Channel(Y, "
                                f"alpha={cfg.alpha}, gamma)"))
    for d in cfg.detectors:
        rows.append((f"Receiver: {DETECTORS[d]}", _receiver_call(cfg, d)))
    return rows
