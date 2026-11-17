"""
General Link Simulator (Streamlit).

Every implemented block is selectable and connected end to end through
Integration/link_engine.py:

    Constellation (PAM / QAM / PSK / ASK / custom)
    -> coding scheme (SISO, Repetition, Alamouti, V-BLAST, DPSK, differential Alamouti, FSK)
    -> waveform per antenna (SC, SC + CP, OFDM)
    -> channel (AWGN, fast Rayleigh / Rician, block MIMO Rayleigh / Rician, multipath MIMO,
       legacy OFDM link) + noise (Gaussian or alpha-stable)
    -> every receiver detector that applies to that combination.

Layout
------
* Sidebar: the link configuration (transmitter, channel, receiver).
* Main area: simulation settings and the results, organised in tabs.

Simulation kinds
----------------
* Single point     one noise level.
* SNR sweep        BER / SER versus Es/N0 or Eb/N0.
* Parameter sweep  BER / SER / goodput versus one link parameter (Nr, Nt, M, L, cyclic
                   prefix, subcarriers, Rician K, repetition factor or alpha) at a fixed SNR.
                   Every value is simulated with the same seed and frame count; values that
                   the modules do not support are skipped and reported with the reason.

Run
---
From the project root (the folder that contains Integration/, Transmitter/, ...):

    streamlit run General_Link_Simulator.py

The file also works as a page of a multipage app placed in a pages/ folder.
"""

import csv
import dataclasses
import hashlib
import io
import math
import os
import sys
import time

# Make the project packages importable whether this file sits in the project root or in a
# pages/ sub-folder of it.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.dirname(_HERE)):
    if os.path.isdir(os.path.join(_cand, "Integration")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import streamlit as st

from Integration import link_engine as le

st.set_page_config(page_title="General Link Simulator", page_icon="📡", layout="wide")

BER_FLOOR = 1e-8
ORDERS = {"PAM": [2, 4, 8, 16, 32, 64, 128, 256], "PSK": [2, 4, 8, 16, 32, 64, 128, 256],
          "ASK": [2, 4, 8, 16], "QAM": [4, 16, 64, 256, 1024]}
FSK_ORDERS = [2, 4, 8, 16, 32, 64]
DEFAULT_CUSTOM = "00: 1+1j, 01: -1+1j, 11: -1-1j, 10: 1-1j"
MAX_SWEEP_VALUES = 25
SIM_KINDS = ["Single point", "SNR sweep", "Parameter sweep"]


# ============================================================
# Helpers
# ============================================================
def skey(name, options=()):
    """Widget key that changes when the option list changes (avoids stale selections)."""
    sig = hashlib.md5("|".join(map(str, options)).encode()).hexdigest()[:8]
    return f"gl_{name}_{sig}"


def pick(label, options, table, name, index=0, help=None):
    return st.selectbox(label, options, index=min(index, len(options) - 1),
                        format_func=lambda k: table[k], key=skey(name, options), help=help)


def clamp(x):
    """Value for a log axis: NaN stays NaN (gap), zero errors are drawn at BER_FLOOR."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return max(float(x), BER_FLOOR)


def to_csv(data):
    """dict of equal-length columns -> CSV text (labels are quoted where needed)."""
    keys = list(data)
    n = len(data[keys[0]]) if keys else 0
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(keys)
    for i in range(n):
        writer.writerow(["" if data[k][i] is None else data[k][i] for k in keys])
    return buf.getvalue()


def config_summary(cfg):
    if cfg.scheme == "FSK":
        mod = f"{cfg.M}-FSK"
    elif cfg.constellation == "CUSTOM":
        mod = f"custom mapping (M = {cfg.M})"
    else:
        mod = f"{cfg.M}-{cfg.constellation}"
    parts = [le.SCHEMES[cfg.scheme].split(" (")[0], mod, le.WAVEFORMS[cfg.waveform],
             le.CHANNELS[cfg.channel].split(" (")[0],
             f"Nt = {le.n_tx_of(cfg)}, Nr = {cfg.n_rx}", le.NOISES[cfg.noise].split(" (")[0]]
    return " · ".join(parts)


# ============================================================
# Parameter sweep: which parameters, which values
# ============================================================
# key -> (label, type, default values text). M is handled with a multiselect.
SWEEP_PARAMS = {
    "n_rx": ("Receive antennas Nr", int, "1, 2, 3, 4"),
    "n_tx": ("Transmit antennas Nt (V-BLAST)", int, "2, 3, 4"),
    "M": ("Modulation order M", int, None),
    "L_taps": ("Number of channel taps L", int, "1, 2, 3, 4, 6, 8"),
    "cyclic_prefix": ("Cyclic prefix N_CP", int, None),
    "n_subcarriers": ("Subcarriers / block size N", int, "16, 32, 64, 128"),
    "K": ("Rician K-factor (linear)", float, "0, 0.5, 1, 3, 10, 30"),
    "rep_L": ("Repetition factor L", int, "1, 2, 3, 4"),
    "alpha": ("Alpha-stable exponent alpha", float, "1.2, 1.4, 1.6, 1.8, 2.0"),
}


def sweepable_params(cfg):
    """Parameters that exist for the current configuration."""
    out = ["n_rx"]
    if cfg.scheme == "VBLAST":
        out.append("n_tx")
    if cfg.constellation != "CUSTOM" or cfg.scheme == "FSK":
        out.append("M")
    if cfg.channel in ("MULTIPATH", "LEGACY_OFDM"):
        out.append("L_taps")
    if cfg.waveform in ("SC_CP", "OFDM"):
        out += ["cyclic_prefix", "n_subcarriers"]
    if cfg.channel in ("RICIAN_FAST", "RICIAN_BLOCK"):
        out.append("K")
    if cfg.scheme == "REPETITION":
        out.append("rep_L")
    if cfg.noise == "ALPHA_STABLE":
        out.append("alpha")
    return out


def m_options(cfg):
    if cfg.scheme == "FSK":
        return FSK_ORDERS
    if cfg.scheme in ("DPSK", "DIFF_ALAMOUTI"):
        return ORDERS["PSK"]
    return ORDERS.get(cfg.constellation, [])


def parse_values(text, typ):
    """Comma-separated numbers -> sorted list without duplicates (raises ValueError)."""
    vals = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
        except ValueError:
            raise ValueError(f"'{part}' is not a number.") from None
        if not math.isfinite(v):
            raise ValueError(f"'{part}' is not a finite number.")
        if typ is int:
            if v != int(v):
                raise ValueError(f"'{part}' must be an integer.")
            v = int(v)
        vals.append(v)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("Enter at least one value.")
    if len(vals) > MAX_SWEEP_VALUES:
        raise ValueError(f"Enter at most {MAX_SWEEP_VALUES} values.")
    return vals


def check_values(param, vals, cfg):
    """Basic range checks; combination checks are left to link_engine.validate."""
    lo = {"n_rx": 1, "n_tx": 2, "L_taps": 1, "cyclic_prefix": 0, "n_subcarriers": 2,
          "K": 0.0, "rep_L": 1}
    hi = {"n_rx": 8, "n_tx": 8, "L_taps": 16, "rep_L": 16}
    for v in vals:
        if param in lo and v < lo[param]:
            raise ValueError(f"{SWEEP_PARAMS[param][0]} must be at least {lo[param]}.")
        if param in hi and v > hi[param]:
            raise ValueError(f"{SWEEP_PARAMS[param][0]} must be at most {hi[param]}.")
        if param == "alpha" and not 0 < v <= 2:
            raise ValueError("alpha must be in (0, 2].")
        if param == "cyclic_prefix" and v >= int(cfg.n_subcarriers):
            raise ValueError(f"The cyclic prefix must be smaller than N = {cfg.n_subcarriers}.")
        if param == "n_subcarriers" and v <= int(cfg.cyclic_prefix):
            raise ValueError(f"N must be larger than the cyclic prefix ({cfg.cyclic_prefix}).")


def run_parameter_sweep(cfg, param, values, n_bits, n_frames, definition, snr_db, seed,
                        progress=None):
    """
    Simulate the link once per parameter value (same seed, same number of frames).
    The frame length is rounded to the nearest valid multiple for every value, and
    detectors that do not apply to a value are left out for that value only.
    """
    dets = list(cfg.detectors)
    rows = []
    for i, v in enumerate(values):
        row = {"value": v, "ok": False}
        try:
            cv = dataclasses.replace(cfg, **{param: v})
            cv.detectors = []                                   # own list, never shared
            c = le.make_constellation(cv)
            allowed = le.detectors_for(cv, c)
            use = [d for d in dets if d in allowed]
            if not use:
                raise ValueError("none of the selected detectors applies to this value")
            cv.detectors = use
            le.validate(cv)
            mult = le.bit_multiple(cv)
            nb = max(mult, int(round(n_bits / mult)) * mult)
            N0 = le.n0_from_snr(cv, snr_db, nb, definition)
            link = le.Link(cv)
            r = link.simulate(nb, N0, n_frames, seed=seed)
            theory = link.theory_ser([N0])
            row.update(ok=True, bits=nb, N0=r["N0"], gamma=r["gamma"], rate=r["rate"],
                       EsN0_dB=r["EsN0_dB"], EbN0_dB=r["EbN0_dB"],
                       per_detector=r["per_detector"],
                       missing=[d for d in dets if d not in use],
                       theory_ser=None if theory is None else theory[0])
        except Exception as exc:                              # report, keep sweeping
            row["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if progress is not None:
            progress((i + 1) / len(values), v)
    return rows


# ============================================================
# SIDEBAR: link configuration
# ============================================================
cfg = le.LinkConfig()

with st.sidebar:
    st.header("Link configuration")
    try:  # only resolvable when this file runs as a page of the multipage app
        st.page_link("controllertestv1.py", label="Back to the classic simulator", icon="⬅️")
    except Exception:
        pass

    # ---------------- 1 · Transmitter ----------------
    st.subheader("1 · Transmitter")
    cfg.scheme = pick("Coding / transmission scheme", list(le.SCHEMES), le.SCHEMES, "scheme",
                      help="Coding_Schemes.py: Repetition, Alamouti, V-BLAST, DifferentialPSK, "
                           "DifferentialAlamouti. FSK uses receiver.fsk_encode.")
    if cfg.scheme == "FSK":
        cfg.constellation = "PAM"
        cfg.M = st.selectbox("Number of tones M", FSK_ORDERS, index=1, key=skey("fsk_M"))
        st.caption("FSK uses M orthogonal tones; only the Gray bit labels of the "
                   "constellation are used.")
    else:
        cons_opts = (["PSK", "CUSTOM"] if cfg.scheme in ("DPSK", "DIFF_ALAMOUTI")
                     else list(le.CONSTELLATIONS))
        cfg.constellation = pick("Constellation", cons_opts, le.CONSTELLATIONS, "cons",
                                 help="ASK / custom mappings use Constellation(..., mapping=dict). "
                                      "Energy detection needs points of distinct magnitude (ASK).")
        if cfg.constellation == "CUSTOM":
            text = st.text_area("Mapping 'label: symbol' (2^k labels of k bits)",
                                value=DEFAULT_CUSTOM, key=skey("custom"))
            try:
                cfg.custom_mapping = le.parse_mapping(text)
                cfg.M = len(cfg.custom_mapping)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
        else:
            orders = ORDERS[cfg.constellation]
            cfg.M = st.selectbox("Modulation order M", orders,
                                 index=1 if len(orders) > 1 else 0,
                                 key=skey("M", (cfg.constellation,) + tuple(orders)))
    cfg.A = st.number_input("Average symbol energy A", min_value=1e-6, value=1.0, step=0.5,
                            key=skey("A"))

    if cfg.scheme == "VBLAST":
        cfg.n_tx = int(st.number_input("Transmit antennas Nt", min_value=2, max_value=8,
                                       value=2, step=1, key=skey("nt")))
    elif cfg.scheme in ("ALAMOUTI", "DIFF_ALAMOUTI"):
        cfg.n_tx = 2
        st.caption("Nt = 2 (fixed by the code).")
    else:
        cfg.n_tx = 1
    if cfg.scheme == "REPETITION":
        cfg.rep_L = int(st.number_input("Repetition factor L", min_value=1, max_value=16,
                                        value=2, step=1, key=skey("repL")))
    if le.n_tx_of(cfg) > 1:
        cfg.power_split = st.checkbox(
            "Power split 1/√Nt (total energy A per channel use)", value=True,
            key=skey("ps"),
            help="Unchecked: every antenna sends energy A (total Nt·A), as the original "
                 "controller's spatial multiplexing mode does.")

    cfg.waveform = pick("Waveform (applied per antenna)", le.waveforms_for(cfg.scheme),
                        le.WAVEFORMS, "wf" + cfg.scheme)
    if cfg.waveform in ("SC_CP", "OFDM"):
        c1, c2 = st.columns(2)
        with c1:
            cfg.n_subcarriers = int(st.number_input(
                "Subcarriers N", min_value=2, value=64, step=1, key=skey("nsc"),
                help="OFDM size, or block size for SC with a cyclic prefix."))
        with c2:
            cfg.cyclic_prefix = int(st.number_input(
                "Cyclic prefix N_CP", min_value=0, max_value=max(0, cfg.n_subcarriers - 1),
                value=min(16, cfg.n_subcarriers - 1), step=1, key=skey("cp")))

    # ---------------- 2 · Channel ----------------
    st.divider()
    st.subheader("2 · Channel")
    ch_opts = le.channels_for(cfg.scheme, cfg.waveform)
    cfg.channel = pick("Channel model", ch_opts, le.CHANNELS, "ch")
    # The key depends on Nt, so Nr starts at Nt again whenever Nt changes (e.g. when
    # switching to V-BLAST or Alamouti); otherwise an old Nr = 1 would silently hide ZF.
    cfg.n_rx = int(st.number_input("Receive antennas Nr", min_value=1, max_value=8,
                                   value=max(1, le.n_tx_of(cfg)), step=1,
                                   key=skey("nr", (le.n_tx_of(cfg),))))
    if cfg.channel in ("RICIAN_FAST", "RICIAN_BLOCK"):
        cfg.K = st.number_input("Rician K-factor (linear)", min_value=0.0, value=1.0, step=0.5,
                                key=skey("K"))
    if cfg.channel == "RICIAN_BLOCK":
        cfg.los = pick("LoS matrix", list(le.LOS_MODELS), le.LOS_MODELS, "los")
    if cfg.channel in ("MULTIPATH", "LEGACY_OFDM"):
        cfg.L_taps = int(st.number_input("Number of taps L (uniform power-delay profile)",
                                         min_value=1, max_value=16, value=3, step=1,
                                         key=skey("L")))
    if cfg.channel == "LEGACY_OFDM":
        cfg.legacy_block = int(st.number_input("Legacy OFDM block length", min_value=2,
                                               value=64, step=1, key=skey("lblk")))
        st.caption("OFDMChannel.OFDM does its own IDFT, CP (= L), channel and DFT, so the "
                   "transmitter waveform is plain SC here (no OFDM twice).")
    noise_opts = le.noises_for(cfg.channel)
    cfg.noise = pick("Noise", noise_opts, le.NOISES, "noise")
    if not le.ALPHA_STABLE_AVAILABLE:
        st.caption("alpha-stable noise needs the aub_htp package (not installed).")
    if cfg.noise == "ALPHA_STABLE":
        cfg.alpha = st.slider("Characteristic exponent alpha", min_value=0.5, max_value=2.0,
                              value=1.8, step=0.05, key=skey("alpha"))

    # ---------------- 3 · Receiver ----------------
    st.divider()
    st.subheader("3 · Receiver")
    try:
        det_opts = le.detectors_for(cfg)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if not det_opts:
        st.error("No detector applies to this combination.")
        st.stop()
    cfg.detectors = st.multiselect(
        "Detectors to compare (same frames for all)", det_opts, default=det_opts,
        format_func=lambda k: le.DETECTORS[k], key=skey("det", det_opts))
    if any(d.startswith("MLSE") for d in cfg.detectors):
        with st.expander("MLSE options (Receiver.mlse_detect_mimo)"):
            cfg.mlse_n_keep = int(st.number_input("M-algorithm survivors n_keep", min_value=1,
                                                  value=16, step=1, key=skey("nkeep")))
            nc = int(st.number_input("K-best sphere candidates n_cand (0 = off)", min_value=0,
                                     value=0, step=1, key=skey("ncand")))
            cfg.mlse_n_cand = nc or None
            sh = int(st.number_input("Channel-shortening prefilter taps (0 = off)", min_value=0,
                                     max_value=cfg.L_taps, value=0, step=1, key=skey("short")))
            cfg.mlse_shorten = sh or None
    if cfg.channel == "MULTIPATH" and cfg.waveform == "SC":
        states, Mv = le.viterbi_states(cfg)
        st.caption(f"Full Viterbi trellis: {states:,} states × {Mv} branches"
                   + ("" if le.viterbi_fits(cfg) else " — too large, so it is not offered."))
    st.caption("Coherent detectors are given the exact channel realization (perfect CSI); "
               "differential, FSK and energy detectors use no channel knowledge.")


# ============================================================
# MAIN AREA: header and validation
# ============================================================
st.title("📡 General Link Simulator")
st.caption(config_summary(cfg))

try:
    le.validate(cfg)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Signal chain of the current configuration (module function per block)"):
    chain_now = le.chain_description(cfg)
    st.table({"Block": [b for b, _ in chain_now], "Call": [c for _, c in chain_now]})

# ============================================================
# SIMULATION SETTINGS
# ============================================================
st.subheader("Simulation")
mult = le.bit_multiple(cfg)
s1, s2, s3, s4 = st.columns(4)
with s1:
    kind = st.radio("Simulation kind", SIM_KINDS, key=skey("kind"))
with s2:
    default_bits = mult * max(1, int(round(1024 / mult)))
    n_bits = int(st.number_input(f"Bits per frame (multiple of {mult})", min_value=mult,
                                 value=default_bits, step=mult, key=skey("nbits", (mult,))))
with s3:
    fading = cfg.channel != "AWGN"
    n_frames = int(st.number_input(
        "Frames (independent channel realizations)" if fading else "Frames",
        min_value=1, value=100 if fading else 10, step=10, key=skey("frames", (fading,))))
with s4:
    fixed_seed = st.checkbox("Reproducible (fixed seed)", value=True, key=skey("fixseed"))
    seed = int(st.number_input("Seed", min_value=0, value=7, step=1,
                               key=skey("seed"))) if fixed_seed else None

if n_bits % mult:
    st.error(f"The frame length must be a multiple of {mult} bits for this configuration.")
    st.stop()

N0 = gamma = None
definition = "EsN0"
snr_db = None
sweep_param = None
sweep_values = None

if kind == "Single point":
    if cfg.noise == "GAUSSIAN":
        n1, n2 = st.columns(2)
        with n1:
            spec = st.radio("Noise level given as", ["Es/N0 [dB]", "Eb/N0 [dB]", "N0 (linear)"],
                            horizontal=True, key=skey("spec"))
        with n2:
            if spec == "N0 (linear)":
                N0 = st.number_input("N0", min_value=0.0, value=0.1, step=0.01, format="%.6f",
                                     key=skey("N0"))
            else:
                snr_in = st.number_input(spec, value=10.0, step=1.0, key=skey("snr", (spec,)))
                N0 = le.n0_from_snr(cfg, snr_in, n_bits,
                                    "EsN0" if spec.startswith("Es") else "EbN0")
    else:
        gamma = st.number_input("Noise scale gamma", min_value=0.0, value=0.1, step=0.01,
                                format="%.4f", key=skey("gamma"))
        N0 = 4 * gamma ** 2
        st.caption("For alpha < 2 the noise variance is infinite; detectors that need a "
                   "noise level use the alpha = 2 equivalent N0 = 4·gamma².")

elif kind == "SNR sweep":
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        definition = pick("SNR definition", list(le.SNR_DEFINITIONS), le.SNR_DEFINITIONS, "def")
    with w2:
        snr_min = st.number_input("SNR min [dB]", value=0.0, step=1.0, key=skey("smin"))
    with w3:
        snr_max = st.number_input("SNR max [dB]", value=20.0, step=1.0, key=skey("smax"))
    with w4:
        n_pts = int(st.number_input("Points", min_value=2, max_value=101, value=11, step=1,
                                    key=skey("npts")))
    if snr_max <= snr_min:
        st.error("SNR max must be greater than SNR min.")
        st.stop()
    if cfg.noise == "ALPHA_STABLE":
        st.caption("alpha-stable sweep: the Gaussian-equivalent SNR is mapped to "
                   "gamma = sqrt(N0 / 4).")

else:  # Parameter sweep
    params = sweepable_params(cfg)
    p1, p2 = st.columns([1, 2])
    with p1:
        sweep_param = st.selectbox("Parameter to sweep", params,
                                   format_func=lambda k: SWEEP_PARAMS[k][0],
                                   key=skey("sparam", params))
    label, typ, default_text = SWEEP_PARAMS[sweep_param]
    with p2:
        try:
            if sweep_param == "M":
                opts = m_options(cfg)
                default_m = [m for m in opts if m <= 64][:4] or opts[:1]
                sweep_values = sorted(st.multiselect(
                    "Values of M", opts, default=default_m, key=skey("svals_M", opts)))
                if not sweep_values:
                    raise ValueError("Select at least one value of M.")
            else:
                if sweep_param == "cyclic_prefix":
                    n_sc = int(cfg.n_subcarriers)
                    start = max(0, int(cfg.L_taps) - 1) if cfg.channel == "MULTIPATH" else 0
                    cands = sorted({c for c in (start, start + 2, n_sc // 8, n_sc // 4,
                                                n_sc // 2) if start <= c < n_sc})
                    default_text = ", ".join(str(c) for c in cands)
                text = st.text_input(f"Values of {label} (comma-separated)",
                                     value=default_text,
                                     key=skey("svals_" + sweep_param,
                                              (default_text, cfg.n_subcarriers,
                                               cfg.L_taps, cfg.channel)))
                sweep_values = parse_values(text, typ)
            check_values(sweep_param, sweep_values, cfg)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
    q1, q2 = st.columns([1, 2])
    with q1:
        definition = pick("SNR definition", list(le.SNR_DEFINITIONS), le.SNR_DEFINITIONS,
                          "pdef")
    with q2:
        snr_db = st.number_input("Fixed SNR [dB]", value=10.0, step=1.0, key=skey("psnr"))
    notes = ["Every value is simulated with the same seed and number of frames.",
             "The frame length is rounded to the nearest valid multiple for every value.",
             "Detectors that do not apply to a value are left out for that value only."]
    if sweep_param == "cyclic_prefix" and cfg.channel == "MULTIPATH":
        notes.append(f"The modules require N_CP ≥ L − 1 = {int(cfg.L_taps) - 1} on the "
                     "multipath channel; smaller values are reported as skipped.")
    if sweep_param in ("cyclic_prefix", "n_subcarriers", "rep_L") and definition == "EsN0":
        notes.append("With Es/N0 the cyclic-prefix and repetition energy is not charged to the "
                     "bits; choose Eb/N0 to include that overhead.")
    if cfg.noise == "ALPHA_STABLE":
        notes.append("alpha-stable noise: the SNR is mapped to gamma = sqrt(N0 / 4).")
    st.caption(" ".join(notes))

# ============================================================
# RUN
# ============================================================
if st.button("Run simulation", type="primary"):
    try:
        t0 = time.perf_counter()
        base = {"cfg": dataclasses.replace(cfg, detectors=list(cfg.detectors)),
                "n_bits": n_bits, "n_frames": n_frames, "seed": seed,
                "chain": le.chain_description(cfg)}
        if kind == "Parameter sweep":
            bar = st.progress(0.0)
            rows = run_parameter_sweep(
                cfg, sweep_param, sweep_values, n_bits, n_frames, definition, snr_db, seed,
                progress=lambda f, v: bar.progress(f, text=f"{SWEEP_PARAMS[sweep_param][0]}"
                                                           f" = {v}"))
            bar.empty()
            payload = dict(base, kind="param", rows=rows, param=sweep_param,
                           definition=definition, snr_db=snr_db)
        else:
            link = le.Link(cfg)
            if kind == "Single point":
                result = link.simulate(n_bits, N0, n_frames, gamma=gamma, seed=seed)
                payload = dict(base, kind="point", result=result)
            else:
                bar = st.progress(0.0)
                snrs = list(np.linspace(snr_min, snr_max, n_pts))
                result = link.sweep(n_bits, snrs, n_frames, definition, seed=seed,
                                    progress=lambda f, s: bar.progress(f, text=f"SNR = {s:.2f} dB"))
                bar.empty()
                payload = dict(base, kind="sweep", result=result, definition=definition)
            payload.update(table=link.c.table.copy(), mapping=dict(link.c.mapping))
        payload["elapsed"] = time.perf_counter() - t0
        st.session_state["gl_payload"] = payload
    except Exception as exc:          # show every module error in the GUI
        st.error(f"Simulation failed: {type(exc).__name__}: {exc}")


# ============================================================
# RESULT VIEWS
# ============================================================
def bit_table(bits, decoded, names, n_show=48):
    n_show = min(n_show, len(bits))
    cell = "padding:4px 6px;border:1px solid #888;text-align:center;"
    html = ["<div style='overflow-x:auto'><table style='border-collapse:collapse;"
            "white-space:nowrap'><tr><th style='" + cell + "'>Row</th>"]
    html += [f"<th style='{cell}'>{j + 1}</th>" for j in range(n_show)] + ["</tr>"]
    html += [f"<tr><td style='{cell}'>Transmitted</td>"]
    html += [f"<td style='{cell}'>{int(b)}</td>" for b in bits[:n_show]] + ["</tr>"]
    for name in names:
        html.append(f"<tr><td style='{cell}'>{name}</td>")
        for b, d in zip(bits[:n_show], decoded[name][:n_show]):
            color = "#15803d" if b == d else "#dc2626"
            html.append(f"<td style='{cell}background:{color};color:white;"
                        f"font-weight:bold'>{int(d)}</td>")
        html.append("</tr>")
    html.append("</table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def constellation_plot(table, mapping):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(table.real, table.imag, s=40, color="#1f77b4", zorder=3)
    if len(table) <= 64:
        for label, s in mapping.items():
            ax.annotate(label, (s.real, s.imag), textcoords="offset points", xytext=(4, 4),
                        fontsize=7)
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Constellation (Gray labels)", fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def show_channel(H, channel):
    H = np.asarray(H)
    if channel in ("AWGN",) + le.BLOCK and H.ndim == 2:
        st.write("Channel matrix H of the last frame (|H|, rows = receive antennas):")
        st.dataframe(np.round(np.abs(H), 4))
    elif channel == "MULTIPATH" and H.ndim == 3:
        rows = [{"r": r + 1, "t": t + 1, "tap l": l, "|h|": float(abs(H[r, t, l])),
                 "phase (rad)": float(np.angle(H[r, t, l]))}
                for r in range(H.shape[0]) for t in range(H.shape[1]) for l in range(H.shape[2])]
        st.write("Channel taps of the last frame:")
        st.dataframe(rows)
    elif channel in le.FAST:
        st.write(f"Per-sample fading: mean |h|² over the last frame = "
                 f"{float(np.mean(np.abs(H) ** 2)):.3f} (expected 1).")
    else:
        st.write("No channel view is available for this channel model.")


def signal_chain_view(chain):
    st.table({"Block": [b for b, _ in chain], "Call": [c for _, c in chain]})


def error_rate_plot(x, curves, xlabel, ylabel, title, theory=None, log2x=False, logy=True):
    fig, ax = plt.subplots(figsize=(6, 4))
    plot = ax.semilogy if logy else ax.plot
    for name, ys in curves.items():
        plot(x, [clamp(y) for y in ys] if logy else ys, marker="o", label=name)
    if theory is not None:
        plot(x, [clamp(y) for y in theory], "k--", label="Theory (receiver.theory_ser_qam)")
    if log2x:
        ax.set_xscale("log", base=2)
    if log2x or len(x) <= 12:                 # ticks only at the simulated values
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.xaxis.set_minor_locator(NullLocator())   # x only; keep the log-y minor grid
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    # Legend below the axes so that it never hides a curve (st.pyplot crops tightly).
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False)
    st.pyplot(fig)
    plt.close(fig)


def results_point(payload):
    r, pcfg = payload["result"], payload["cfg"]
    names = list(r["per_detector"])
    t_err, t_bits, t_ch, t_const, t_chain = st.tabs(
        ["Error rates", "Bits", "Channel", "Constellation", "Signal chain"])
    with t_err:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Es/N0", f"{r['EsN0_dB']:.2f} dB")
        m2.metric("Eb/N0", f"{r['EbN0_dB']:.2f} dB")
        m3.metric("Rate", f"{r['rate']:.3f} bits / channel use")
        m4.metric("N0" if pcfg.noise == "GAUSSIAN" else "gamma",
                  f"{r['N0']:.4g}" if pcfg.noise == "GAUSSIAN" else f"{r['gamma']:.4g}")
        st.caption(f"Channel uses per frame and antenna (incl. CP / reference symbols): "
                   f"{r['channel_uses_per_frame']}. Goodput = rate × (1 − BER).")
        st.dataframe({
            "Detector": [le.DETECTORS[d] for d in names],
            "BER": [v["ber"] for v in r["per_detector"].values()],
            "SER": [v["ser"] for v in r["per_detector"].values()],
            "Bit errors": [v["bit_errors"] for v in r["per_detector"].values()],
            "Bits": [v["bits"] for v in r["per_detector"].values()],
            "Goodput [bits/use]": [v["goodput"] for v in r["per_detector"].values()],
        })
    with t_bits:
        st.write("Transmitted and decoded bits (last frame, first 48 bits):")
        last = r["last"]
        bit_table(last["bits"], {le.DETECTORS[d]: last["decoded"][d] for d in names},
                  [le.DETECTORS[d] for d in names])
    with t_ch:
        show_channel(r["last"]["H"], pcfg.channel)
    with t_const:
        if pcfg.scheme == "FSK":
            st.write("FSK uses orthogonal tones; there is no signal-space constellation.")
        else:
            constellation_plot(payload["table"], payload["mapping"])
    with t_chain:
        signal_chain_view(payload["chain"])


def results_snr_sweep(payload):
    r, pcfg = payload["result"], payload["cfg"]
    snr = r["snr_db"]
    xlabel = "Es/N0 [dB]" if payload["definition"] == "EsN0" else "Eb/N0 [dB]"
    t_err, t_data, t_const, t_chain = st.tabs(
        ["Error rates", "Data", "Constellation", "Signal chain"])
    with t_err:
        p1, p2 = st.columns(2)
        for col, metric, title in ((p1, "ber", "BER"), (p2, "ser", "SER")):
            with col:
                curves = {le.DETECTORS[d]: v[metric] for d, v in r["per_detector"].items()}
                theory = r.get("theory_ser") if metric == "ser" else None
                error_rate_plot(snr, curves, xlabel, title, f"{title} vs SNR", theory=theory)
        st.caption(f"Values at or below {BER_FLOOR:.0e} are drawn at {BER_FLOOR:.0e} "
                   "(zero observed errors).")
    with t_data:
        data = {xlabel: snr, "N0": r["N0"]}
        for d, v in r["per_detector"].items():
            data[f"BER {le.DETECTORS[d]}"] = v["ber"]
            data[f"SER {le.DETECTORS[d]}"] = v["ser"]
        if r.get("theory_ser"):
            data["SER theory"] = r["theory_ser"]
        st.dataframe(data)
        st.download_button("Download sweep data (CSV)", to_csv(data), "snr_sweep.csv",
                           "text/csv")
    with t_const:
        if pcfg.scheme == "FSK":
            st.write("FSK uses orthogonal tones; there is no signal-space constellation.")
        else:
            constellation_plot(payload["table"], payload["mapping"])
    with t_chain:
        signal_chain_view(payload["chain"])


def results_param_sweep(payload):
    rows, param, pcfg = payload["rows"], payload["param"], payload["cfg"]
    label = SWEEP_PARAMS[param][0]
    ok = [row for row in rows if row["ok"]]
    skipped = [row for row in rows if not row["ok"]]
    snr_name = "Es/N0" if payload["definition"] == "EsN0" else "Eb/N0"
    dets = list(pcfg.detectors)

    t_err, t_data, t_chain = st.tabs(["Error rates", "Data", "Signal chain"])
    with t_err:
        st.caption(f"{label} swept at {snr_name} = {payload['snr_db']:g} dB, "
                   f"{payload['n_frames']} frames per value.")
        if skipped:
            st.warning("Skipped values: " + "; ".join(
                f"{row['value']} ({row['reason']})" for row in skipped))
        if not ok:
            st.error("No value of the sweep could be simulated.")
        else:
            x = [row["value"] for row in ok]

            def series(metric):
                out = {}
                for d in dets:
                    ys = [row["per_detector"][d][metric] if d in row["per_detector"]
                          else float("nan") for row in ok]
                    if not all(isinstance(y, float) and math.isnan(y) for y in ys):
                        out[le.DETECTORS[d]] = ys
                return out

            theory = [row["theory_ser"] for row in ok]
            theory = None if any(t is None for t in theory) else theory
            log2x = param == "M"
            p1, p2 = st.columns(2)
            with p1:
                error_rate_plot(x, series("ber"), label, "BER", f"BER vs {label}", log2x=log2x)
            with p2:
                error_rate_plot(x, series("ser"), label, "SER", f"SER vs {label}",
                                theory=theory, log2x=log2x)
            g1, _ = st.columns(2)
            with g1:
                error_rate_plot(x, series("goodput"), label, "Goodput [bits / channel use]",
                                f"Goodput vs {label}", log2x=log2x, logy=False)
            missing = [f"{le.DETECTORS[d]} at {label} = "
                       + ", ".join(str(row["value"]) for row in ok if d in row["missing"])
                       for d in dets if any(d in row["missing"] for row in ok)]
            if missing:
                st.info("Not available (left out): " + "; ".join(missing) + ".")
            st.caption(f"Values at or below {BER_FLOOR:.0e} are drawn at {BER_FLOOR:.0e} "
                       "(zero observed errors). Goodput = rate × (1 − BER).")
    with t_data:
        data = {label: [row["value"] for row in rows],
                "Simulated": ["yes" if row["ok"] else "no" for row in rows],
                "Bits per frame": [row.get("bits") for row in rows],
                "N0": [row.get("N0") for row in rows],
                "Es/N0 [dB]": [row.get("EsN0_dB") for row in rows],
                "Eb/N0 [dB]": [row.get("EbN0_dB") for row in rows],
                "Rate [bits/use]": [row.get("rate") for row in rows]}
        for d in dets:
            for metric, name in (("ber", "BER"), ("ser", "SER"), ("goodput", "Goodput")):
                data[f"{name} {le.DETECTORS[d]}"] = [
                    row["per_detector"][d][metric]
                    if row["ok"] and d in row["per_detector"] else None for row in rows]
        if any(row.get("theory_ser") is not None for row in rows):
            data["SER theory"] = [row.get("theory_ser") for row in rows]
        data["Note"] = [row.get("reason", "") for row in rows]
        st.dataframe(data)
        st.download_button("Download sweep data (CSV)", to_csv(data),
                           f"parameter_sweep_{param}.csv", "text/csv")
    with t_chain:
        st.caption("Signal chain of the base configuration (before the swept parameter is "
                   "changed).")
        signal_chain_view(payload["chain"])


# ============================================================
# RESULTS
# ============================================================
payload = st.session_state.get("gl_payload")
if payload:
    pcfg = payload["cfg"]
    st.divider()
    st.subheader("Results")
    kind_label = {"point": "Single point", "sweep": "SNR sweep", "param": "Parameter sweep"}
    st.caption(f"{kind_label[payload['kind']]} · {config_summary(pcfg)} · "
               f"{payload['n_bits']} bits × {payload['n_frames']} frames · "
               f"seed {payload['seed'] if payload['seed'] is not None else 'random'} · "
               f"{payload['elapsed']:.2f} s")
    current = dataclasses.replace(cfg, detectors=list(cfg.detectors))
    if current != pcfg:
        st.info("The configuration in the sidebar has changed since these results were "
                "computed. Press Run simulation to update them.")
    if payload["kind"] == "point":
        results_point(payload)
    elif payload["kind"] == "sweep":
        results_snr_sweep(payload)
    else:
        results_param_sweep(payload)
