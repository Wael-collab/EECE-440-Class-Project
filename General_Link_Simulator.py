"""
General Link Simulator (Streamlit page of controllertestv1.py).

Every implemented block is selectable and connected end to end through
Integration/link_engine.py:

    Constellation (PAM / QAM / PSK / ASK / custom)
    -> coding scheme (SISO, Repetition, Alamouti, V-BLAST, DPSK, differential Alamouti, FSK)
    -> waveform per antenna (SC, SC + CP, OFDM)
    -> channel (AWGN, fast Rayleigh / Rician, block MIMO Rayleigh / Rician, multipath MIMO,
       legacy OFDM link) + noise (Gaussian or alpha-stable)
    -> every receiver detector that applies to that combination.

Run the app from the project root:  streamlit run controllertestv1.py
"""

import hashlib
import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from Integration import link_engine as le

st.set_page_config(page_title="General Link Simulator", page_icon="📡", layout="wide")
st.title("📡 General Link Simulator")

try:  # only resolvable when the page runs inside the multipage app
    st.page_link("controllertestv1.py", label="Back to the classic simulator", icon="⬅️")
except Exception:
    pass

BER_FLOOR = 1e-8
ORDERS = {"PAM": [2, 4, 8, 16, 32, 64, 128, 256], "PSK": [2, 4, 8, 16, 32, 64, 128, 256],
          "ASK": [2, 4, 8, 16], "QAM": [4, 16, 64, 256, 1024]}
FSK_ORDERS = [2, 4, 8, 16, 32, 64]
DEFAULT_CUSTOM = "00: 1+1j, 01: -1+1j, 11: -1-1j, 10: 1-1j"


def skey(name, options=()):
    """Widget key that changes when the option list changes (avoids stale selections)."""
    sig = hashlib.md5("|".join(map(str, options)).encode()).hexdigest()[:8]
    return f"gl_{name}_{sig}"


def pick(label, options, table, name, index=0, help=None):
    return st.selectbox(label, options, index=min(index, len(options) - 1),
                        format_func=lambda k: table[k], key=skey(name, options), help=help)


# ============================================================
# CONFIGURATION
# ============================================================
col_tx, col_ch, col_rx = st.columns(3)
cfg = le.LinkConfig()

with col_tx:
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
                "Subcarriers / block size N", min_value=2, value=64, step=1, key=skey("nsc")))
        with c2:
            cfg.cyclic_prefix = int(st.number_input(
                "Cyclic prefix N_CP", min_value=0, max_value=max(0, cfg.n_subcarriers - 1),
                value=min(16, cfg.n_subcarriers - 1), step=1, key=skey("cp")))

with col_ch:
    st.subheader("2 · Channel")
    ch_opts = le.channels_for(cfg.scheme, cfg.waveform)
    cfg.channel = pick("Channel model", ch_opts, le.CHANNELS, "ch")
    cfg.n_rx = int(st.number_input("Receive antennas Nr", min_value=1, max_value=8,
                                   value=max(1, le.n_tx_of(cfg)), step=1, key=skey("nr")))
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

with col_rx:
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

try:
    le.validate(cfg)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Signal chain (which module function every block calls)", expanded=False):
    st.table({"Block": [b for b, _ in le.chain_description(cfg)],
              "Call": [c for _, c in le.chain_description(cfg)]})

# ============================================================
# SIMULATION SETTINGS
# ============================================================
st.subheader("4 · Simulation")
mult = le.bit_multiple(cfg)
s1, s2, s3, s4 = st.columns(4)
with s1:
    kind = st.radio("Simulation kind", ["Single point", "BER sweep"], horizontal=True,
                    key=skey("kind"))
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
                snr_db = st.number_input(spec, value=10.0, step=1.0, key=skey("snr", (spec,)))
                N0 = le.n0_from_snr(cfg, snr_db, n_bits,
                                    "EsN0" if spec.startswith("Es") else "EbN0")
        gamma = None
    else:
        gamma = st.number_input("Noise scale gamma", min_value=0.0, value=0.1, step=0.01,
                                format="%.4f", key=skey("gamma"))
        N0 = 4 * gamma ** 2
        st.caption("For alpha < 2 the noise variance is infinite; detectors that need a "
                   "noise level use the alpha = 2 equivalent N0 = 4·gamma².")
else:
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        definition = pick("SNR definition", list(le.SNR_DEFINITIONS), le.SNR_DEFINITIONS, "def")
    with w2:
        snr_min = st.number_input("SNR min [dB]", value=0.0, step=1.0, key=skey("smin"))
    with w3:
        snr_max = st.number_input("SNR max [dB]", value=20.0, step=1.0, key=skey("smax"))
    with w4:
        n_pts = int(st.number_input("Points", min_value=2, value=11, step=1, key=skey("npts")))
    if snr_max <= snr_min:
        st.error("SNR max must be greater than SNR min.")
        st.stop()
    if cfg.noise == "ALPHA_STABLE":
        st.caption("alpha-stable sweep: the Gaussian-equivalent SNR is mapped to "
                   "gamma = sqrt(N0 / 4).")

# ============================================================
# RUN
# ============================================================
if st.button("Run simulation", type="primary"):
    try:
        link = le.Link(cfg)
        t0 = time.perf_counter()
        if kind == "Single point":
            result = link.simulate(n_bits, N0, n_frames, gamma=gamma, seed=seed)
            payload = {"kind": "point", "result": result}
        else:
            bar = st.progress(0.0)
            snrs = list(np.linspace(snr_min, snr_max, n_pts))
            result = link.sweep(n_bits, snrs, n_frames, definition, seed=seed,
                                progress=lambda f, s: bar.progress(f, text=f"SNR = {s:.2f} dB"))
            bar.empty()
            payload = {"kind": "sweep", "result": result, "definition": definition}
        payload.update(elapsed=time.perf_counter() - t0, cfg=cfg, n_bits=n_bits,
                       n_frames=n_frames, chain=le.chain_description(cfg),
                       table=link.c.table.copy(), mapping=dict(link.c.mapping),
                       theory=link.theory_available())
        st.session_state["gl_payload"] = payload
    except Exception as exc:          # show every module error in the GUI
        st.error(f"Simulation failed: {type(exc).__name__}: {exc}")


# ============================================================
# RESULTS
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
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
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


payload = st.session_state.get("gl_payload")
if payload:
    pcfg = payload["cfg"]
    st.divider()
    st.subheader("Results")
    st.caption(f"{le.SCHEMES[pcfg.scheme]} · {le.WAVEFORMS[pcfg.waveform]} · "
               f"{le.CHANNELS[pcfg.channel]} · Nt = {le.n_tx_of(pcfg)}, Nr = {pcfg.n_rx} · "
               f"{payload['n_bits']} bits × {payload['n_frames']} frames · "
               f"{payload['elapsed']:.2f} s")
    r = payload["result"]

    if payload["kind"] == "point":
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Es/N0", f"{r['EsN0_dB']:.2f} dB")
        m2.metric("Eb/N0", f"{r['EbN0_dB']:.2f} dB")
        m3.metric("Rate", f"{r['rate']:.3f} bits / channel use")
        m4.metric("N0" if pcfg.noise == "GAUSSIAN" else "gamma",
                  f"{r['N0']:.4g}" if pcfg.noise == "GAUSSIAN" else f"{r['gamma']:.4g}")
        st.caption(f"Channel uses per frame and antenna (incl. CP / reference symbols): "
                   f"{r['channel_uses_per_frame']}. Goodput = rate × (1 − BER).")
        names = list(r["per_detector"])
        st.dataframe({
            "Detector": [le.DETECTORS[d] for d in names],
            "BER": [v["ber"] for v in r["per_detector"].values()],
            "SER": [v["ser"] for v in r["per_detector"].values()],
            "Bit errors": [v["bit_errors"] for v in r["per_detector"].values()],
            "Bits": [v["bits"] for v in r["per_detector"].values()],
            "Goodput [bits/use]": [v["goodput"] for v in r["per_detector"].values()],
        })
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("Transmitted and decoded bits (last frame, first 48 bits):")
            last = r["last"]
            bit_table(last["bits"], {le.DETECTORS[d]: last["decoded"][d] for d in names},
                      [le.DETECTORS[d] for d in names])
            show_channel(last["H"], pcfg.channel)
        with c2:
            if pcfg.scheme != "FSK":
                constellation_plot(payload["table"], payload["mapping"])
    else:
        snr = r["snr_db"]
        xlabel = "Es/N0 [dB]" if payload["definition"] == "EsN0" else "Eb/N0 [dB]"
        p1, p2 = st.columns(2)
        for col, metric, title in ((p1, "ber", "BER"), (p2, "ser", "SER")):
            with col:
                fig, ax = plt.subplots(figsize=(6, 4))
                for d, v in r["per_detector"].items():
                    ax.semilogy(snr, [max(x, BER_FLOOR) for x in v[metric]], marker="o",
                                label=le.DETECTORS[d])
                if metric == "ser" and r.get("theory_ser"):
                    ax.semilogy(snr, [max(x, BER_FLOOR) for x in r["theory_ser"]], "k--",
                                label="Theory (receiver.theory_ser_qam)")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(title)
                ax.set_title(f"{title} vs SNR")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(fontsize=7)
                st.pyplot(fig)
                plt.close(fig)
        st.caption(f"Values at or below {BER_FLOOR:.0e} are clamped for the log axis "
                   "(zero observed errors).")
        data = {xlabel: snr, "N0": r["N0"]}
        for d, v in r["per_detector"].items():
            data[f"BER {le.DETECTORS[d]}"] = v["ber"]
            data[f"SER {le.DETECTORS[d]}"] = v["ser"]
        if r.get("theory_ser"):
            data["SER theory"] = r["theory_ser"]
        st.dataframe(data)
        buf = io.StringIO()
        keys = list(data)
        buf.write(",".join(keys) + "\n")
        for i in range(len(snr)):
            buf.write(",".join(str(data[k][i]) for k in keys) + "\n")
        st.download_button("Download sweep data (CSV)", buf.getvalue(), "sweep.csv", "text/csv")

    with st.expander("Signal chain used for these results"):
        st.table({"Block": [b for b, _ in payload["chain"]],
                  "Call": [c for _, c in payload["chain"]]})
