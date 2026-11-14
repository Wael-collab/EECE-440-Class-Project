import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from Transmitter.Transmitter import Constellation, Transmitter
from Channel.AWGNChannel import AWGN_Channel
from Channel.MIMORicianFlatChannel import MIMORicianFlatChannel
from Receiver.receiver import Receiver


st.set_page_config(
    page_title="Communication Systems Simulator",
    page_icon="📡",
    layout="wide",
)
st.title("📡 Communication Systems Simulator")
st.write("Welcome to the EECE 440 CSS Simulator!")


# ============================================================
# COMMON FUNCTIONS
# ============================================================

MODES = [
    "Transmit Diversity",
    "Receive Diversity",
    "Spatial Multiplexing (ZF)",
]

BER_FLOOR = 1e-8  # used only for plotting zero-error points on log axes

# Multi-tap (frequency-selective) channel
MULTITAP = "Multi-Tap Channel"
SISO_ISI_MODE = "SISO (ISI equalization)"
VITERBI = "Viterbi (MLSE)"
# Equalizer name -> (kind, structure) for Receiver.detect_sc; None = Viterbi.
EQUALIZERS = {
    VITERBI: None,
    "Linear MMSE": ("mmse", "linear"),
    "Linear ZF": ("zf", "linear"),
    "DFE (MMSE)": ("mmse", "dfe"),
    "DFE (ZF)": ("zf", "dfe"),
}
DEFAULT_EQUALIZERS = [VITERBI, "Linear MMSE", "Linear ZF", "DFE (MMSE)"]
TAP_MODELS = ["Rayleigh fading taps", "Fixed taps"]
DEFAULT_FIXED_TAPS = "0.407, 0.815, 0.407"   # Proakis channel B
# Same trellis limits as Receiver.mlse_detect_mimo (max_states and the
# S * M * Nr <= 2^22 check), so the dialog rejects what the receiver would.
MAX_TRELLIS_STATES = 1 << 16
MAX_TRELLIS_BRANCHES = 1 << 22


def modulation_orders(constellation):
    if constellation == "PAM":
        return [2, 4, 8, 16, 32, 64, 128, 256]
    if constellation == "QAM":
        return [4, 16, 64, 256, 1024]
    return [2, 4, 8, 16, 32, 64, 128, 256]


def check_mimo_mode(channel):
    """Validate the antenna constraints for the selected mode/channel."""
    mode = st.session_state.mimo_mode
    numtx = st.session_state.numtx
    numrx = st.session_state.numrx

    # AWGN is locked to Spatial Multiplexing (ZF).
    if channel == "AWGN" and mode != "Spatial Multiplexing (ZF)":
        st.error("AWGN supports only Spatial Multiplexing (ZF).")
        return False

    if mode == "Receive Diversity" and numrx < 2:
        st.error("Receive Diversity requires Nr >= 2.")
        return False

    if mode == "Transmit Diversity" and numtx < 2:
        st.error("Transmit Diversity requires Nt >= 2.")
        return False

    # Preserve the existing ZF restriction on fading channels.
    if (
        mode == "Spatial Multiplexing (ZF)"
        and channel in ("Rician", "Rayleigh")
        and numrx < numtx
    ):
        st.error("Zero-forcing detection requires Nr >= Nt.")
        return False

    return True


def display_bit_comparison(bits, decoded_bits, num_rows, N, row_label="Transmitter",
                           row_names=None):
    """
    Display sent/decoded bits; rows can be transmitters or receivers.
    row_names (optional) replaces the default "<row_label> i" row titles.
    """
    if row_names is None:
        row_names = [f"{row_label} {i+1}" for i in range(num_rows)]
    st.subheader("Transmitted and Received Bits")
    display_bits = min(int(N), 500)
    st.caption(
        f"Displaying the first {display_bits} bits per {row_label.lower()} "
        f"(out of {N} bits)."
    )

    table = (
        "<div style='overflow-x:auto;width:100%;'>"
        "<table style='border-collapse:collapse;white-space:nowrap;width:max-content;'>"
        "<tr><th style='padding:8px;border:1px solid #888;'>"
        f"{row_label}</th>"
        "<th style='padding:8px;border:1px solid #888;'>Signal</th>"
    )
    for j in range(display_bits):
        table += f"<th style='padding:8px;border:1px solid #888;'>{j+1}</th>"
    table += "</tr>"

    for i in range(num_rows):
        table += (
            "<tr><td rowspan='2' style='padding:8px;border:1px solid #888;'>"
            f"{row_names[i]}</td>"
            "<td style='padding:8px;border:1px solid #888;'>Transmitted</td>"
        )
        for j in range(display_bits):
            table += (
                "<td style='padding:8px;border:1px solid #888;text-align:center;'>"
                f"{int(bits[i][j])}</td>"
            )
        table += (
            "</tr><tr><td style='padding:8px;border:1px solid #888;'>Received</td>"
        )
        for j in range(display_bits):
            original = int(bits[i][j])
            decoded = int(decoded_bits[i][j])
            color = "#15803d" if original == decoded else "#dc2626"
            table += (
                "<td style='padding:8px;border:1px solid #888;text-align:center;"
                f"background-color:{color};color:white;font-weight:bold;'>"
                f"{decoded}</td>"
            )
        table += "</tr>"

    table += "</table></div>"
    st.markdown(table, unsafe_allow_html=True)


def display_timing_breakdown(timings, caption):
    st.subheader("Simulation Execution Time")
    for operation, elapsed_time in timings.items():
        st.write(f"**{operation}:** {elapsed_time:.6f} seconds")
    st.caption(caption)


def channel_prefix(channel):
    """Session-state key prefix for each channel's settings and results."""
    if channel == "AWGN":
        return ""
    if channel == MULTITAP:
        return "multitap_"
    return f"{channel.lower()}_"


def clear_session_keys(keys):
    for key in keys:
        st.session_state.pop(key, None)


def result_keys(prefix):
    names = [
        "bits", "samples", "received", "decoded_bits", "ber", "H",
        "error_counts", "total_bits", "sweep_results",
    ]
    keys = [f"{prefix}{name}" for name in names]
    keys.append("simulation_timings" if not prefix else f"{prefix}timings")
    return keys


def clear_simulation_results():
    clear_session_keys(result_keys(""))
    clear_session_keys(result_keys("rician_"))
    clear_session_keys(result_keys("rayleigh_"))
    clear_session_keys(result_keys("multitap_"))
    clear_session_keys([
        "multitap_transmitters", "multitap_receivers", "multitap_c",
        "multitap_constellation_type", "multitap_M_value",
        "multitap_A_value", "multitap_N_value", "multitap_N0_value",
        "multitap_repetitions_value", "multitap_tap_model_value",
        "multitap_L_value", "multitap_pdp_value", "multitap_decay_db_value",
        "multitap_fixed_taps_value", "multitap_equalizers_value",
    ])
    clear_session_keys([
        "transmitters", "receivers", "c", "constellation", "M", "A", "N", "N0",
        "rician_transmitters", "rician_receivers", "rician_c",
        "rician_constellation_type", "rician_M_value", "rician_A_value",
        "rician_N_value", "rician_N0_value", "rician_K_value",
        "rician_K_input", "rician_repetitions",
        "rician_repetitions_value",
        "rayleigh_transmitters", "rayleigh_receivers", "rayleigh_c",
        "rayleigh_constellation_type", "rayleigh_M_value",
        "rayleigh_A_value", "rayleigh_N_value", "rayleigh_N0_value",
        "rayleigh_repetitions", "rayleigh_repetitions_value",
        "sweep_snr_min", "sweep_snr_max", "sweep_num_points",
        "awgn_repetitions_value",
    ])


def los_matrix(numrx, numtx):
    """Keep the same deterministic LoS phase model as the original code."""
    return np.exp(
        1j * 2 * np.pi
        * np.outer(np.arange(numrx), np.arange(numtx)) / numrx
    )


def fading_channel(symbols, N0, numtx, numrx, K, H_los):
    """Use the existing channel implementation; Rayleigh is simply K=0."""
    received, H = MIMORicianFlatChannel(
        symbols=symbols,
        Nzero=N0,
        Nt=numtx,
        Nr=numrx,
        K=K,
        H_los=H_los,
    )
    return np.asarray(received), np.asarray(H)


def decoded_vector(raw_bits, N):
    """Normalize the output shape of Receiver.decode/decode_mrc."""
    result = np.asarray(raw_bits, dtype=int).reshape(-1)
    if result.size != N:
        raise ValueError(
            f"Receiver returned {result.size} bits; expected {N}. "
            "Check the receiver decoder and channel array shapes."
        )
    return result


# ============================================================
# MULTI-TAP (FREQUENCY-SELECTIVE) CHANNEL
# ============================================================

def tap_powers(L, pdp="Uniform", decay_db=0.0):
    """Average tap powers E|h_l|^2, normalized so they sum to 1."""
    L = int(L)
    if pdp == "Exponential":
        p = 10.0 ** (-float(decay_db) * np.arange(L) / 10.0)
    else:
        p = np.ones(L)
    return p / p.sum()


def parse_taps(text):
    """Read comma-separated tap coefficients and scale them to unit energy."""
    parts = [
        part.strip() for part in str(text).replace(";", ",").split(",")
        if part.strip()
    ]
    if len(parts) < 2:
        raise ValueError(
            "Enter at least 2 tap coefficients, separated by commas."
        )
    try:
        taps = np.array([
            complex(part.replace(" ", "").replace("i", "j")) for part in parts
        ])
    except ValueError:
        raise ValueError(
            "Could not read the tap coefficients; use numbers such as "
            "0.8 or 0.3-0.1j."
        ) from None
    energy = float(np.sum(np.abs(taps) ** 2))
    if not np.isfinite(energy) or energy == 0:
        raise ValueError("The taps must be finite and not all zero.")
    return taps / np.sqrt(energy)


def format_complex(z):
    z = complex(z)
    if abs(z.imag) < 1e-12:
        return f"{z.real:.4g}"
    return f"{z.real:.3f}{z.imag:+.3f}j"


def draw_multitap_taps():
    """Channel taps h (L,) for one realization, from the dialog settings."""
    state = st.session_state
    if state.multitap_tap_model_value == "Fixed taps":
        return np.asarray(state.multitap_fixed_taps_value, dtype=complex)

    powers = tap_powers(
        state.multitap_L_value,
        state.multitap_pdp_value,
        state.multitap_decay_db_value,
    )
    gaussian = (
        np.random.randn(powers.size) + 1j * np.random.randn(powers.size)
    )
    return np.sqrt(powers / 2.0) * gaussian


def multitap_channel(symbols, taps, N0):
    """
    y[n] = sum_l h[l] x[n-l] + w[n], n = 0 .. T + L - 2.

    The full convolution is kept, including the L - 1 tail samples, so the
    Viterbi detector can use them (mlse_detect(..., tail=True)). Noise is
    added with the same AWGN_Channel as the other channels, so N0 means
    the same thing everywhere.
    """
    clean = np.convolve(np.asarray(symbols).ravel(), np.asarray(taps))
    return np.asarray(AWGN_Channel(clean[None, :], N0)).reshape(-1)


def detect_multitap(receiver, equalizer, received, taps, N0):
    """
    Detected symbol indices (T,) from one equalizer.

    The receiver is handed the taps of the CURRENT realization, so every
    equalizer is redesigned for each new channel (perfect CSI):
        Viterbi (MLSE)          -> Receiver.mlse_detect
        Linear / DFE, ZF / MMSE -> Receiver.detect_sc (time-domain FIR
                                   equalizer designed for these taps)
    received is the full convolution output (T + L - 1 samples), hence
    tail=True for both.
    """
    spec = EQUALIZERS[equalizer]
    if spec is None:
        return receiver.mlse_detect(received, taps, tail=True)
    kind, structure = spec
    return receiver.detect_sc(
        received, taps, N0=N0, tail=True, kind=kind, structure=structure,
    )


def ber_from_detected_symbols(receiver, detected, tx_bits):
    """
    Turn an equalizer's detected symbols into bits and count bit errors.

    detected: symbol indices, as returned by Receiver.mlse_detect and
              Receiver.detect_sc (table index = bit label). Complex or real symbol values
              are also accepted and are sliced to the nearest
              constellation point first.
    tx_bits:  the transmitted bits of the same frame.

    Returns (decoded_bits, n_errors, ber).
    """
    detected = np.asarray(detected).ravel()
    if not np.issubdtype(detected.dtype, np.integer):
        detected = receiver._slice(detected)

    decoded_bits = receiver.bit_table[detected].ravel()
    tx_bits = np.asarray(tx_bits, dtype=int).ravel()
    if decoded_bits.size != tx_bits.size:
        raise ValueError(
            f"Equalizer returned {decoded_bits.size} bits; "
            f"expected {tx_bits.size}."
        )

    n_errors = int(np.count_nonzero(decoded_bits != tx_bits))
    return decoded_bits, n_errors, n_errors / tx_bits.size


def run_multitap_simulation(transmitters, receivers, N, N0, repetitions=1):
    """
    SISO single-carrier link over a frequency-selective (multi-tap)
    channel, equalized with each selected equalizer.

    For each realization:
        bits -> symbols x (T,) -> new taps h -> y = h * x + noise
        (length T + L - 1) -> for every selected equalizer:
        detect_multitap(receiver, eq, y, h, N0) -> bits -> bit errors

    The taps stay fixed over a frame and are redrawn every realization
    (block fading). The receiver is given the new taps every time, and all
    equalizers see exactly the same bits, taps and noise, so their BERs
    are directly comparable. BER = total errors / total bits per equalizer.
    """
    N = int(N)
    repetitions = int(repetitions)
    transmitter, receiver = transmitters[0], receivers[0]
    equalizers = list(st.session_state.multitap_equalizers_value)
    n_eq = len(equalizers)

    k = int(receiver.k)
    if N % k:
        raise ValueError(f"N={N} must be divisible by k={k}.")
    num_symbols = N // k

    total_errors = np.zeros(n_eq, dtype=np.int64)
    decoded = np.zeros((n_eq, N), dtype=int)
    timings = {
        "Bit generation": 0.0,
        "Transmitter modulation": 0.0,
        "Channel tap generation": 0.0,
        "Channel transmission (convolution + noise)": 0.0,
    }
    for eq in equalizers:
        timings[f"Detection: {eq}"] = 0.0
    timings["BER calculation"] = 0.0
    simulation_start = time.perf_counter()

    for _ in range(repetitions):
        start = time.perf_counter()
        bits = np.random.randint(0, 2, size=N)
        timings["Bit generation"] += time.perf_counter() - start

        start = time.perf_counter()
        symbols = np.asarray(transmitter.run(bits, N)).ravel()
        if symbols.size != num_symbols:
            raise ValueError(
                f"Transmitter returned {symbols.size} symbols; "
                f"expected {num_symbols}."
            )
        timings["Transmitter modulation"] += time.perf_counter() - start

        start = time.perf_counter()
        taps = draw_multitap_taps()          # new h for this realization
        timings["Channel tap generation"] += time.perf_counter() - start

        start = time.perf_counter()
        received = multitap_channel(symbols, taps, N0)
        timings["Channel transmission (convolution + noise)"] += (
            time.perf_counter() - start
        )

        for e, eq in enumerate(equalizers):
            start = time.perf_counter()
            detected = detect_multitap(receiver, eq, received, taps, N0)
            timings[f"Detection: {eq}"] += time.perf_counter() - start

            start = time.perf_counter()
            decoded[e], n_errors, _ = ber_from_detected_symbols(
                receiver, detected, bits
            )
            total_errors[e] += n_errors
            timings["BER calculation"] += time.perf_counter() - start

    timings["Total simulation time"] = time.perf_counter() - simulation_start

    total_bits = np.full(n_eq, repetitions * N, dtype=np.int64)
    return {
        "bits": np.tile(bits, (n_eq, 1)),
        "samples": symbols[None, :],
        "received": received[None, :],
        "decoded_bits": decoded.copy(),
        "ber": total_errors / total_bits,
        "H": taps,
        "timings": timings,
        "error_counts": total_errors,
        "total_bits": total_bits,
    }


def is_fixed_multitap(channel):
    """True for the multi-tap channel with fixed taps (one deterministic channel)."""
    return (
        channel == MULTITAP
        and st.session_state.get("multitap_tap_model_value") == "Fixed taps"
    )


def multitap_summary():
    """One-line description of the configured multi-tap channel."""
    state = st.session_state
    L = int(state.multitap_L_value)
    M = int(state.multitap_M_value)
    if state.multitap_tap_model_value == "Fixed taps":
        taps = ", ".join(
            format_complex(t) for t in state.multitap_fixed_taps_value
        )
        channel = f"fixed taps h = [{taps}] (unit energy)"
    else:
        if state.multitap_pdp_value == "Exponential":
            profile = (
                f"exponential power-delay profile, "
                f"{state.multitap_decay_db_value:g} dB/tap"
            )
        else:
            profile = "uniform power-delay profile"
        channel = (
            f"{L} Rayleigh fading taps, {profile} (unit total power)"
        )
    equalizers = list(state.multitap_equalizers_value)
    text = f"Channel: {channel}. Equalizers: {', '.join(equalizers)}"
    if VITERBI in equalizers:
        text += f" (Viterbi trellis: {M ** (L - 1):,} states)"
    return text + "."


def display_multitap_taps(taps):
    """Table of the channel taps used in the last realization."""
    taps = np.asarray(taps, dtype=complex).ravel()
    state = st.session_state
    st.subheader(
        "Channel Taps" if state.multitap_tap_model_value == "Fixed taps"
        else "Channel Taps (last realization)"
    )
    table = {
        "Tap l": list(range(taps.size)),
        "h_l": [format_complex(t) for t in taps],
        "|h_l|^2": [float(abs(t) ** 2) for t in taps],
    }
    if state.multitap_tap_model_value != "Fixed taps":
        table["Average power E|h_l|^2"] = list(tap_powers(
            state.multitap_L_value,
            state.multitap_pdp_value,
            state.multitap_decay_db_value,
        ))
    st.dataframe(table)


def run_simulation(channel, transmitters, receivers, N, N0, repetitions=1, K=0.0):
    """
    Simulation modes:

    Receive Diversity (Nr >= 2):
        One physical transmitter is active at a time. For each Tx, its
        own bitstream is sent to all Nr receive antennas and decoded by
        MRC. Each fading realization draws a separate SIMO channel for
        each sequentially active transmitter.

    Transmit Diversity (Nt >= 2, any Nr >= 1):
        Each receiving antenna is assigned its own information stream,
        delivered sequentially as in the original Rx-diversity design.
        All Nt physical Tx antennas transmit that receiver's symbols in
        Nt orthogonal time slots (one active per slot). That receiving
        antenna then combines its Nt observations with MRC. All Nr
        destination streams occupy separate frames, and one channel H
        is held constant throughout a realization. The 1/sqrt(Nt)
        amplitude factor keeps total energy per information symbol
        constant. This is transmit-repetition diversity, NOT Alamouti.

    Spatial Multiplexing (ZF):
        Keep the original simultaneous independent Tx streams and ZF
        receiver. On fading channels, use Receiver.decode_mimo as before.

    For Rayleigh/Rician, BER is computed as total errors / total bits
    over the user-selected number of independent channel realizations.

    Multi-Tap Channel: handled by run_multitap_simulation (SISO,
    Viterbi MLSE equalizer).
    """
    if channel == MULTITAP:
        return run_multitap_simulation(
            transmitters, receivers, N, N0, repetitions
        )

    mode = st.session_state.mimo_mode
    numtx = int(st.session_state.numtx)
    numrx = int(st.session_state.numrx)
    N = int(N)
    repetitions = int(repetitions)

    k = int(receivers[0].k)
    if N % k:
        raise ValueError(f"N={N} must be divisible by k={k}.")
    num_symbols = N // k

    is_tx_diversity = mode == "Transmit Diversity"
    num_results = numrx if is_tx_diversity else numtx
    total_errors = np.zeros(num_results, dtype=np.int64)
    total_bits = np.zeros(num_results, dtype=np.int64)

    timings = {
        "Bit generation": 0.0,
        "Transmitter modulation": 0.0,
        "Channel matrix / LoS setup": 0.0,
        "Channel transmission": 0.0,
        "Coherent detection and decoding": 0.0,
        "BER calculation": 0.0,
    }
    simulation_start = time.perf_counter()

    last_bits = None
    last_samples = None
    last_received = None
    last_decoded_bits = None
    last_H = None

    for _ in range(repetitions):
        # ----------------------------------------------------
        # 1. Generate information streams
        # ----------------------------------------------------
        start = time.perf_counter()
        if is_tx_diversity:
            bits = np.random.randint(0, 2, size=(numrx, N))
        else:
            bits = np.random.randint(0, 2, size=(numtx, N))
        timings["Bit generation"] += time.perf_counter() - start

        # ----------------------------------------------------
        # 2. Modulate information streams
        # ----------------------------------------------------
        start = time.perf_counter()
        stream_count = numrx if is_tx_diversity else numtx
        samples = np.asarray([
            transmitters[i].run(bits[i], N) for i in range(stream_count)
        ])
        if samples.shape != (stream_count, num_symbols):
            raise ValueError(
                f"Unexpected transmitter output shape {samples.shape}; "
                f"expected ({stream_count}, {num_symbols})."
            )
        timings["Transmitter modulation"] += time.perf_counter() - start

        # ----------------------------------------------------
        # 3. Channel setup
        # ----------------------------------------------------
        start = time.perf_counter()
        H_los = los_matrix(numrx, numtx)
        timings["Channel matrix / LoS setup"] += time.perf_counter() - start

        # ----------------------------------------------------
        # 4. Channel + diversity/MIMO detection
        # ----------------------------------------------------
        if mode == "Receive Diversity":
            received_per_transmitter = []
            channel_per_transmitter = []
            decoded = []

            for i in range(numtx):
                start = time.perf_counter()
                if channel == "AWGN":
                    h = np.ones(numrx, dtype=complex)
                    tx_replicated = np.tile(samples[i], (numrx, 1))
                    Y = np.asarray(AWGN_Channel(tx_replicated, N0))
                else:
                    Y, H_i = fading_channel(
                        symbols=samples[i:i+1],
                        N0=N0,
                        numtx=1,
                        numrx=numrx,
                        K=K,
                        H_los=H_los[:, i:i+1],
                    )
                    h = H_i[:, 0]

                if Y.shape != (numrx, num_symbols):
                    raise ValueError(f"SIMO channel returned shape {Y.shape}.")
                timings["Channel transmission"] += time.perf_counter() - start

                start = time.perf_counter()
                decoded.append(decoded_vector(
                    receivers[i].decode_mrc(Y=Y, H=h), N
                ))
                timings["Coherent detection and decoding"] += (
                    time.perf_counter() - start
                )
                received_per_transmitter.append(Y)
                channel_per_transmitter.append(h)

            received = np.asarray(received_per_transmitter)
            H = np.asarray(channel_per_transmitter)

        elif mode == "Transmit Diversity":
            start = time.perf_counter()
            frame_symbols = numtx * num_symbols
            X = np.zeros((numtx, numrx * frame_symbols), dtype=complex)

            for r in range(numrx):
                for t in range(numtx):
                    begin = r * frame_symbols + t * num_symbols
                    X[t, begin:begin + num_symbols] = (
                        samples[r] / np.sqrt(numtx)
                    )

            if channel == "AWGN":
                H = np.ones((numrx, numtx), dtype=complex)
                Y = np.asarray(AWGN_Channel(H @ X, N0))
            else:
                Y, H = fading_channel(
                    symbols=X,
                    N0=N0,
                    numtx=numtx,
                    numrx=numrx,
                    K=K,
                    H_los=H_los,
                )

            if Y.shape != (numrx, numrx * frame_symbols):
                raise ValueError(
                    f"Transmit-diversity channel returned {Y.shape}."
                )
            timings["Channel transmission"] += time.perf_counter() - start

            start = time.perf_counter()
            received = np.asarray([
                Y[r, r * frame_symbols:(r + 1) * frame_symbols].reshape(
                    numtx, num_symbols
                )
                for r in range(numrx)
            ])

            decoded = [
                decoded_vector(
                    receivers[r].decode_mrc(
                        Y=received[r],
                        H=H[r, :] / np.sqrt(numtx),
                    ),
                    N,
                )
                for r in range(numrx)
            ]
            timings["Coherent detection and decoding"] += (
                time.perf_counter() - start
            )

        else:  # Spatial Multiplexing (ZF)
            start = time.perf_counter()
            if channel == "AWGN":
                H = H_los / np.sqrt(numrx)
                received = np.asarray(AWGN_Channel(H @ samples, N0))
            else:
                received, H = fading_channel(
                    symbols=samples,
                    N0=N0,
                    numtx=numtx,
                    numrx=numrx,
                    K=K,
                    H_los=H_los,
                )
            timings["Channel transmission"] += time.perf_counter() - start

            start = time.perf_counter()
            if channel == "AWGN":
                detected = np.linalg.pinv(H) @ received
                decoded = [
                    decoded_vector(receivers[i].decode(detected[i]), N)
                    for i in range(numtx)
                ]
            else:
                decoded_interleaved = receivers[0].decode_mimo(
                    Y=received,
                    H=H,
                    method="zf",
                    power_split=False,
                )
                decoded = np.asarray(decoded_interleaved).reshape(
                    num_symbols, numtx, k
                ).transpose(1, 0, 2).reshape(numtx, N)

            timings["Coherent detection and decoding"] += (
                time.perf_counter() - start
            )

        decoded = np.asarray(decoded, dtype=int)
        if decoded.shape != (num_results, N):
            raise ValueError(
                f"Decoded bit shape {decoded.shape}; "
                f"expected ({num_results}, {N})."
            )

        # ----------------------------------------------------
        # 5. BER accumulation across independent realizations
        # ----------------------------------------------------
        start = time.perf_counter()
        total_errors += np.count_nonzero(bits != decoded, axis=1)
        total_bits += N
        timings["BER calculation"] += time.perf_counter() - start

        last_bits = bits
        last_samples = samples
        last_received = received
        last_decoded_bits = decoded
        last_H = H

    timings["Total simulation time"] = time.perf_counter() - simulation_start

    return {
        "bits": last_bits,
        "samples": last_samples,
        "received": last_received,
        "decoded_bits": last_decoded_bits,
        "ber": total_errors / total_bits,
        "H": last_H,
        "timings": timings,
        "error_counts": total_errors,
        "total_bits": total_bits,
    }


def run_sweep(channel, transmitters, receivers, N, A, snr_db_list,
              repetitions=1, K=0.0):
    """
    Run the same simulation across a user-supplied Es/N0 sweep.

    SNR convention: SNR = Es / N0, with Es = A (average symbol energy)
    and N0 the channel noise PSD. For each SNR point we set
    N0 = A / 10^(SNR_dB / 10).

    Returns (snr_db_list, ber_per_stream, ber_avg):
        ber_per_stream -> list (one entry per result row) of BER lists
        ber_avg        -> BER list averaged across result rows
    """
    mode = st.session_state.mimo_mode
    numtx = int(st.session_state.numtx)
    numrx = int(st.session_state.numrx)

    if channel == MULTITAP:
        num_results = len(st.session_state.multitap_equalizers_value)
    else:
        num_results = numrx if mode == "Transmit Diversity" else numtx
    ber_per_stream = [[] for _ in range(num_results)]
    ber_avg = []

    progress_bar = st.progress(0.0)
    status = st.empty()

    for idx, snr_db in enumerate(snr_db_list):
        status.text(
            f"Simulating SNR = {snr_db:.2f} dB "
            f"({idx + 1}/{len(snr_db_list)})..."
        )
        snr_linear = 10.0 ** (snr_db / 10.0)
        N0 = A / snr_linear

        result = run_simulation(
            channel=channel,
            transmitters=transmitters,
            receivers=receivers,
            N=N,
            N0=N0,
            repetitions=repetitions,
            K=K,
        )

        for i in range(num_results):
            ber_per_stream[i].append(float(result["ber"][i]))
        ber_avg.append(float(np.mean(result["ber"])))

        progress_bar.progress((idx + 1) / len(snr_db_list))

    progress_bar.empty()
    status.empty()

    return list(snr_db_list), ber_per_stream, ber_avg


def store_results(prefix, results):
    for name in (
        "bits", "samples", "received", "decoded_bits", "ber",
        "H", "error_counts", "total_bits",
    ):
        st.session_state[f"{prefix}{name}"] = results[name]

    timing_key = "simulation_timings" if not prefix else f"{prefix}timings"
    st.session_state[timing_key] = results["timings"]


def initialize_modems(channel):
    prefix = channel_prefix(channel)
    tx_key = f"{prefix}transmitters"
    rx_key = f"{prefix}receivers"

    if tx_key in st.session_state:
        return

    if channel == "AWGN":
        constellation_type = st.session_state.constellation
        M = st.session_state.M
        A = st.session_state.A
    else:
        constellation_type = st.session_state[f"{prefix}constellation_type"]
        M = st.session_state[f"{prefix}M_value"]
        A = st.session_state[f"{prefix}A_value"]

    c = Constellation(constellation_type, M=M, A=A)
    mode = st.session_state.mimo_mode

    logical_tx_count = (
        st.session_state.numrx
        if mode == "Transmit Diversity" else st.session_state.numtx
    )
    decoder_count = (
        st.session_state.numrx
        if mode == "Transmit Diversity" else st.session_state.numtx
    )

    st.session_state[tx_key] = [
        Transmitter(c) for _ in range(logical_tx_count)
    ]
    st.session_state[rx_key] = [
        Receiver(c) for _ in range(decoder_count)
    ]
    st.session_state[f"{prefix}c"] = c

    if mode == "Transmit Diversity":
        st.success(
            f"Initialized {logical_tx_count} sequential receiver-destined "
            f"information streams, using Nt={st.session_state.numtx} "
            "physical antennas for transmit diversity."
        )
    else:
        st.success(
            f"Initialized {logical_tx_count} transmitters and "
            f"{decoder_count} decoders."
        )


def display_simulation_results(channel):
    prefix = channel_prefix(channel)
    state = st.session_state

    if f"{prefix}received" not in state:
        return

    mode = state.mimo_mode
    numtx = state.numtx
    numrx = state.numrx
    N = state.N if channel == "AWGN" else state[f"{prefix}N_value"]

    labels = "Receiver" if mode == "Transmit Diversity" else "Transmitter"
    num_results = numrx if mode == "Transmit Diversity" else numtx
    row_names = None
    if channel == MULTITAP:
        row_names = list(state.multitap_equalizers_value)
        num_results = len(row_names)
        labels = "Equalizer"
    else:
        row_names = [f"{labels} {i+1}" for i in range(num_results)]

    st.subheader(f"{channel} Simulation Results")

    if channel == "Rician":
        st.write(f"Rician K-factor (linear): {state.rician_K_value}")
    elif channel == "Rayleigh":
        st.write("Channel model: Rayleigh fading (K = 0)")

    if channel != "AWGN":
        st.write(
            "Detection mode: equalizers redesigned for the known taps of "
            "every realization (Perfect CSI)"
            if channel == MULTITAP else "Detection mode: Coherent (Perfect CSI)"
        )
        st.write(
            f"Noise power spectral density (N0): "
            f"{state[f'{prefix}N0_value']:.6f}"
        )
        st.write(
            f"Average symbol energy (A): "
            f"{state[f'{prefix}A_value']:.6f}"
        )
        st.write(
            f"Modulation: {state[f'{prefix}constellation_type']}, "
            f"M = {state[f'{prefix}M_value']}"
        )
        if not is_fixed_multitap(channel):
            st.write(
                "Independent channel realizations: "
                f"{state[f'{prefix}repetitions_value']}"
            )
        st.write(
            f"Total tested bits per {labels.lower()}: "
            f"{state[f'{prefix}total_bits'][0]:,}"
        )

    if mode == "Receive Diversity":
        st.info(
            "Receive Diversity: one transmitter sends its own stream at a "
            "time; all Nr receive branches are combined using MRC."
        )

    elif mode == "Transmit Diversity":
        st.info(
            "Transmit Diversity: each receiver has its own sequential "
            "bitstream. All Nt transmit antennas cooperate using "
            "orthogonal time slots (one antenna active per slot), and "
            "that receiver combines Nt observations using MRC. "
            "Total energy per information symbol is fixed. "
            "Each stream requires Nt time slots per symbol, and the "
            "Nr streams are also time-multiplexed. "
            "The BER rows are per receiver, not per transmitter."
        )

    elif channel == "AWGN" and numrx < numtx:
        st.warning(
            "Nr < Nt: spatial multiplexing with pseudo-inverse ZF is "
            "underdetermined and cannot generally separate the streams."
        )

    elif channel == MULTITAP:
        st.info(
            "Multi-tap channel: one transmitter and one receiver. Each "
            "frame is convolved with the channel taps (intersymbol "
            "interference) and noise is added. The receiver is given the "
            "taps and each selected equalizer recovers the symbols; all "
            "equalizers process the same frames. "
            + multitap_summary()
        )
        display_multitap_taps(state[f"{prefix}H"])

    timing_key = "simulation_timings" if not prefix else f"{prefix}timings"

    display_timing_breakdown(
        state[timing_key],
        "Computational time only; browser rendering is excluded. "
        + (
            "Timings are accumulated across all channel realizations."
            if channel != "AWGN" and not is_fixed_multitap(channel) else ""
        ),
    )

    if channel != "AWGN" and not is_fixed_multitap(channel):
        st.caption(
            "The bit comparison below is from the last channel "
            "realization; the BER values use all channel realizations."
        )

    display_bit_comparison(
        state[f"{prefix}bits"],
        state[f"{prefix}decoded_bits"],
        num_results,
        N,
        row_label=labels,
        row_names=row_names,
    )

    st.subheader(
        "Monte Carlo BER (Averaged over channel realizations)"
        if channel != "AWGN" and not is_fixed_multitap(channel)
        else "Bit Error Rates"
    )

    if channel == MULTITAP:
        columns = st.columns(num_results)
        for i, value in enumerate(state[f"{prefix}ber"]):
            with columns[i]:
                st.metric(f"BER - {row_names[i]}", f"{value:.6f}")
    else:
        for i, value in enumerate(state[f"{prefix}ber"]):
            st.metric(f"BER - {labels} {i+1}", f"{value:.6f}")

        st.metric(
            "Overall Average BER",
            f"{np.mean(state[f'{prefix}ber']):.6f}"
        )

    if channel != "AWGN":
        st.subheader("Error Statistics")
        for i in range(num_results):
            st.write(
                f"**{row_names[i]}:** "
                f"{state[f'{prefix}error_counts'][i]:,} errors out of "
                f"{state[f'{prefix}total_bits'][i]:,} bits"
            )


def display_sweep_results(channel):
    """Render the BER-vs-SNR sweep curve and raw data."""
    prefix = channel_prefix(channel)
    state = st.session_state

    if f"{prefix}sweep_results" not in state:
        return

    results = state[f"{prefix}sweep_results"]
    snr_db = results["snr_db"]
    ber_per_stream = results["ber_per_stream"]
    ber_avg = results["ber_avg"]
    labels = results["labels"]
    elapsed = results["elapsed"]

    mode = state.mimo_mode
    numtx = state.numtx
    numrx = state.numrx

    st.subheader(f"{channel} BER Sweep Results")
    st.write(f"MIMO mode: {mode}")
    st.write(f"Nt = {numtx}, Nr = {numrx}")
    if channel == "Rician":
        st.write(f"Rician K-factor (linear): {state.rician_K_value}")
    elif channel == "Rayleigh":
        st.write("Channel model: Rayleigh fading (K = 0)")
    elif channel == MULTITAP:
        st.write(multitap_summary())

    if channel == "AWGN":
        st.write(
            "Bit realizations per SNR point: "
            f"{state.get('awgn_repetitions_value', 1)}"
        )
        st.write(
            f"Modulation: {state.constellation}, M = {state.M}"
        )
        st.write(f"Average symbol energy (A): {state.A:.6f}")
    else:
        if is_fixed_multitap(channel):
            st.write(
                f"Bits per SNR point: {state[f'{prefix}N_value']:,} "
                "(fixed taps, one frame)"
            )
        else:
            st.write(
                "Channel realizations per SNR point: "
                f"{state[f'{prefix}repetitions_value']}"
            )
        st.write(
            f"Average symbol energy (A): "
            f"{state[f'{prefix}A_value']:.6f}"
        )
        st.write(
            f"Modulation: {state[f'{prefix}constellation_type']}, "
            f"M = {state[f'{prefix}M_value']}"
        )

    st.info(
        "SNR convention: Es/N0 (average symbol energy divided by the "
        "one-sided noise power spectral density). For each sweep point "
        "N0 is set to A / 10^(SNR_dB / 10)."
    )

    # ------------------------------------------------------------
    # BER vs SNR(dB) on a semilogy axis
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    if len(ber_per_stream) == 1:
        ber_plot = [max(b, BER_FLOOR) for b in ber_per_stream[0]]
        ax.semilogy(
            snr_db, ber_plot,
            marker="o", linewidth=2, markersize=8, color="#1f77b4",
        )
    else:
        for i, ber_list in enumerate(ber_per_stream):
            ber_plot = [max(b, BER_FLOOR) for b in ber_list]
            ax.semilogy(
                snr_db, ber_plot,
                marker="o", linewidth=1.8,
                label=labels[i],
            )
        if channel != MULTITAP:
            ber_avg_plot = [max(b, BER_FLOOR) for b in ber_avg]
            ax.semilogy(
                snr_db, ber_avg_plot,
                marker="s", linewidth=2.2, linestyle="--",
                color="black", label="Average BER",
            )
        ax.legend(loc="best")

    ax.set_xlabel("SNR (Es/N0) [dB]")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(f"{channel} - {mode} - BER vs SNR")
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.caption(
        "Values at or below "
        f"{BER_FLOOR:.0e} are clamped for visibility on the log axis "
        "(they typically indicate zero observed errors at that SNR)."
    )
    st.caption(f"Total sweep simulation time: {elapsed:.2f} s")

    # ------------------------------------------------------------
    # Data table
    # ------------------------------------------------------------
    st.subheader("Sweep Data")
    table = {"SNR (Es/N0) [dB]": snr_db}
    for i, ber_list in enumerate(ber_per_stream):
        table[f"{labels[i]} BER"] = ber_list
    if len(ber_per_stream) > 1 and channel != MULTITAP:
        table["Average BER"] = ber_avg
    st.dataframe(table)


# ============================================================
# INITIAL SIMULATION CONFIGURATION
# ============================================================

@st.dialog("Simulation Parameters")
def simulation_popup():
    sweep = st.session_state.get("sweep_mode", False)

    if sweep:
        st.write(
            "Sweep mode selected. Choose the channel and MIMO mode, and "
            "the SNR range will be defined in the next dialog."
        )
    else:
        st.write(
            "Let's get you started! What channel do you wish to test today?"
        )

    channel = st.selectbox(
        "Channel Model",
        ["AWGN", "Rayleigh", "Rician", "OFDM", "Multi-Tap Channel"],
    )

    # AWGN is locked to Spatial Multiplexing (ZF); diversity modes are
    # only meaningful on fading channels.
    if channel == "AWGN":
        st.info(
            "AWGN supports only Spatial Multiplexing (ZF). "
            "Diversity modes apply to fading channels."
        )
        mimo_mode = "Spatial Multiplexing (ZF)"
    elif channel == MULTITAP:
        st.info(
            "The multi-tap channel is simulated with 1 transmitter and "
            "1 receiver. The channel taps and the equalizers (Viterbi, "
            "linear, ZF, DFE) are set in the next dialog."
        )
        mimo_mode = SISO_ISI_MODE
    else:
        mimo_mode = st.selectbox(
            "MIMO Transmission Mode",
            MODES,
            index=2,
            key="initial_mimo_mode",
        )

    # In sweep mode, diversity modes collapse to a single BER curve.
    if channel == MULTITAP:
        numtx = 1
        numrx = 1
    elif sweep and mimo_mode == "Receive Diversity":
        st.info(
            "Sweep mode: Receive Diversity uses exactly 1 transmitter "
            "so that one BER curve is produced."
        )
        numtx = 1
        numrx = st.number_input(
            "Number of Receiving Antennas",
            min_value=2,
            value=2,
            step=1,
            key="initial_numrx_diversity",
        )
    elif sweep and mimo_mode == "Transmit Diversity":
        st.info(
            "Sweep mode: Transmit Diversity uses exactly 1 receiver "
            "so that one BER curve is produced."
        )
        numrx = 1
        numtx = st.number_input(
            "Number of Transmitting Antennas",
            min_value=2,
            value=2,
            step=1,
            key="initial_numtx_diversity",
        )
    else:
        if mimo_mode == "Transmit Diversity":
            numtx = st.number_input(
                "Number of Transmitting Antennas",
                min_value=2,
                value=2,
                step=1,
                key="initial_numtx_diversity",
            )
        else:
            numtx = st.number_input(
                "Number of Transmitting Antennas",
                min_value=1,
                value=1,
                step=1,
                key="initial_numtx_other",
            )

        if mimo_mode == "Receive Diversity":
            numrx = st.number_input(
                "Number of Receiving Antennas",
                min_value=2,
                value=2,
                step=1,
                key="initial_numrx_diversity",
            )
        else:
            numrx = st.number_input(
                "Number of Receiving Antennas",
                min_value=1,
                value=1,
                step=1,
                key="initial_numrx_other",
            )

    if st.button("Confirm Parameters"):
        if mimo_mode == "Receive Diversity" and numrx < 2:
            st.error("Receive Diversity requires Nr >= 2.")
            return

        if mimo_mode == "Transmit Diversity" and numtx < 2:
            st.error("Transmit Diversity requires Nt >= 2.")
            return

        clear_simulation_results()

        clear_session_keys([
            "channel", "numtx", "numrx", "mimo_mode",
            "awgn_configured", "rician_configured", "rayleigh_configured",
            "multitap_configured",
        ])

        st.session_state.channel = channel
        st.session_state.numtx = int(numtx)
        st.session_state.numrx = int(numrx)
        st.session_state.mimo_mode = mimo_mode

        # Close the new-simulation dialog before opening
        # the selected channel's parameter dialog.
        st.session_state.open_new_simulation = False

        st.rerun()


# ============================================================
# MAIN SIMULATION BUTTONS AND CONFIGURATION DISPLAY
# ============================================================

if "initialized" not in st.session_state:
    st.session_state.initialized = True

if "open_new_simulation" not in st.session_state:
    st.session_state.open_new_simulation = False

if "sweep_mode" not in st.session_state:
    st.session_state.sweep_mode = False


col_new, col_sweep = st.columns(2)
with col_new:
    if st.button("New BER Simulation"):
        st.session_state.open_new_simulation = True
        st.session_state.sweep_mode = False
        st.rerun()
with col_sweep:
    if st.button("Sweep BER Simulation"):
        st.session_state.open_new_simulation = True
        st.session_state.sweep_mode = True
        st.rerun()


if "channel" in st.session_state:
    st.subheader("Current Simulation Configuration")

    kind = "BER Sweep" if st.session_state.get("sweep_mode", False) else "Single-Point BER"
    st.write(f"Simulation Kind: {kind}")
    st.write(f"Channel: {st.session_state.channel}")
    st.write(f"Transmitting Antennas: {st.session_state.numtx}")
    st.write(f"Receiving Antennas: {st.session_state.numrx}")
    st.write(f"MIMO Mode: {st.session_state.mimo_mode}")

    if st.button("Close Simulation"):
        clear_simulation_results()

        clear_session_keys([
            "channel", "numtx", "numrx", "mimo_mode",
            "awgn_configured", "rician_configured", "rayleigh_configured",
            "multitap_configured",
            "sweep_mode",
        ])

        st.session_state.open_new_simulation = False

        st.rerun()


# ============================================================
# CHANNEL PARAMETER DIALOGS
# ============================================================

@st.dialog("AWGN Simulation Parameters")
def awgn_popup():
    sweep = st.session_state.get("sweep_mode", False)

    st.write("Please select AWGN Channel parameters")

    constellation = st.selectbox(
        "Constellation Type",
        ["PAM", "QAM", "PSK"]
    )

    M = st.selectbox(
        "Modulation Order (M)",
        modulation_orders(constellation)
    )

    A = st.number_input(
        "Average Symbol Energy (A)",
        min_value=0.000001,
        value=1.0,
        step=0.5,
    )

    st.divider()

    N = st.number_input(
        "Message Length (number of bits per information stream)",
        min_value=1,
        value=64,
        step=1,
    )

    if sweep:
        st.subheader("SNR Sweep Range (Es/N0)")
        col1, col2, col3 = st.columns(3)
        with col1:
            snr_min = st.number_input(
                "SNR Min (dB)",
                value=0.0,
                step=1.0,
                key="awgn_snr_min",
            )
        with col2:
            snr_max = st.number_input(
                "SNR Max (dB)",
                value=20.0,
                step=1.0,
                key="awgn_snr_max",
            )
        with col3:
            snr_points = st.number_input(
                "Number of Points",
                min_value=2,
                value=11,
                step=1,
                key="awgn_snr_points",
            )

        if snr_max <= snr_min:
            st.error("SNR Max must be greater than SNR Min.")
            return

        num_repetitions = st.number_input(
            "Bit Realizations per SNR Point",
            min_value=1,
            value=10,
            step=1,
            help=(
                "Number of independent random bit sequences per SNR "
                "point. Higher values yield smoother BER estimates."
            ),
        )

        st.caption(
            "For each sweep point N0 is computed as A / 10^(SNR_dB / 10) "
            "so that SNR = Es/N0 with Es = A."
        )
    else:
        N0 = st.number_input(
            "Noise Power Spectral Density (N0)",
            min_value=0.0,
            value=0.1,
            step=0.01,
            format="%.6f",
        )
        num_repetitions = 1

    if st.button("Start AWGN Simulation"):
        k = int(np.log2(M))

        if N % k:
            st.error(
                f"Message length must be divisible by {k} for M={M}."
            )
            return

        st.session_state.constellation = constellation
        st.session_state.M = M
        st.session_state.A = A
        st.session_state.N = int(N)

        if sweep:
            st.session_state["awgn_repetitions_value"] = int(num_repetitions)
            st.session_state["sweep_snr_min"] = float(snr_min)
            st.session_state["sweep_snr_max"] = float(snr_max)
            st.session_state["sweep_num_points"] = int(snr_points)
        else:
            st.session_state.N0 = float(N0)

        st.session_state.awgn_configured = True

        clear_session_keys(result_keys(""))
        clear_session_keys(["transmitters", "receivers", "c"])

        st.rerun()


def fading_popup(channel):
    """Shared parameter form for Rician and Rayleigh channels."""

    prefix = f"{channel.lower()}_"
    sweep = st.session_state.get("sweep_mode", False)

    st.write(f"Please select {channel} Channel parameters")

    constellation = st.selectbox(
        "Constellation Type",
        ["PAM", "QAM", "PSK"],
        key=f"{prefix}constellation"
    )

    M = st.selectbox(
        "Modulation Order (M)",
        modulation_orders(constellation),
        key=f"{prefix}M"
    )

    A = st.number_input(
        "Average Symbol Energy (A)",
        min_value=0.000001,
        value=1.0,
        step=0.5,
        key=f"{prefix}A",
    )

    st.divider()

    N = st.number_input(
        "Message Length (number of bits per information stream)",
        min_value=1,
        value=64,
        step=1,
        key=f"{prefix}N",
    )

    if sweep:
        st.subheader("SNR Sweep Range (Es/N0)")
        col1, col2, col3 = st.columns(3)
        with col1:
            snr_min = st.number_input(
                "SNR Min (dB)",
                value=0.0,
                step=1.0,
                key=f"{prefix}snr_min",
            )
        with col2:
            snr_max = st.number_input(
                "SNR Max (dB)",
                value=20.0,
                step=1.0,
                key=f"{prefix}snr_max",
            )
        with col3:
            snr_points = st.number_input(
                "Number of Points",
                min_value=2,
                value=11,
                step=1,
                key=f"{prefix}snr_points",
            )

        if snr_max <= snr_min:
            st.error("SNR Max must be greater than SNR Min.")
            return

        num_repetitions = st.number_input(
            "Channel Realizations per SNR Point",
            min_value=1,
            value=100,
            step=10,
            help=(
                "Each SNR point averages over this many independent "
                "channel realizations."
            ),
        )
        st.caption(
            "For each sweep point N0 is computed as A / 10^(SNR_dB / 10) "
            "so that SNR = Es/N0 with Es = A."
        )
    else:
        N0 = st.number_input(
            "Noise Power Spectral Density (N0)",
            min_value=0.0,
            value=0.1,
            step=0.01,
            format="%.6f",
            key=f"{prefix}N0",
        )
        num_repetitions = st.number_input(
            "Number of Channel Realizations to Average Over",
            min_value=1,
            value=100,
            step=10,
            key=f"{prefix}repetitions",
        )

    K = 0.0

    if channel == "Rician":
        K = st.number_input(
            "Rician K-factor (linear scale)",
            min_value=0.0,
            value=1.0,
            step=0.5,
            key="rician_K_input",
        )

        st.caption(
            "K = 0 is Rayleigh; larger K means a stronger LoS component."
        )

    st.info(
        "Detection mode: Coherent Detection (Perfect Channel Knowledge)."
    )

    if st.button(f"Start {channel} Simulation"):
        k = int(np.log2(M))

        if N % k:
            st.error(
                f"Message length must be divisible by {k} for M={M}."
            )
            return

        st.session_state[f"{prefix}constellation_type"] = constellation
        st.session_state[f"{prefix}M_value"] = M
        st.session_state[f"{prefix}A_value"] = A
        st.session_state[f"{prefix}N_value"] = int(N)
        st.session_state[f"{prefix}repetitions_value"] = int(num_repetitions)

        if channel == "Rician":
            st.session_state.rician_K_value = float(K)

        if sweep:
            st.session_state["sweep_snr_min"] = float(snr_min)
            st.session_state["sweep_snr_max"] = float(snr_max)
            st.session_state["sweep_num_points"] = int(snr_points)
        else:
            st.session_state[f"{prefix}N0_value"] = float(N0)

        st.session_state[f"{prefix}configured"] = True

        clear_session_keys(result_keys(prefix))

        clear_session_keys([
            f"{prefix}transmitters",
            f"{prefix}receivers",
            f"{prefix}c",
        ])

        st.rerun()


@st.dialog("Rician Simulation Parameters")
def rician_popup():
    fading_popup("Rician")


@st.dialog("Rayleigh Simulation Parameters")
def rayleigh_popup():
    fading_popup("Rayleigh")


@st.dialog("Multi-Tap Channel Simulation Parameters")
def multitap_popup():
    """Parameters for the frequency-selective channel and its equalizers."""
    prefix = "multitap_"
    sweep = st.session_state.get("sweep_mode", False)

    st.write("Please select Multi-Tap Channel parameters")

    constellation = st.selectbox(
        "Constellation Type",
        ["PAM", "QAM", "PSK"],
        key=f"{prefix}constellation",
    )

    M = st.selectbox(
        "Modulation Order (M)",
        modulation_orders(constellation),
        key=f"{prefix}M",
    )

    A = st.number_input(
        "Average Symbol Energy (A)",
        min_value=0.000001,
        value=1.0,
        step=0.5,
        key=f"{prefix}A",
    )

    st.divider()

    N = st.number_input(
        "Message Length (number of bits per frame)",
        min_value=1,
        value=64,
        step=1,
        key=f"{prefix}N",
    )

    # ---------------- channel taps ----------------
    st.subheader("Multipath Channel")
    tap_model = st.radio(
        "Tap Model",
        TAP_MODELS,
        horizontal=True,
        key=f"{prefix}tap_model",
        help=(
            "Rayleigh fading taps: new complex Gaussian taps every channel "
            "realization. Fixed taps: the same deterministic ISI channel "
            "every realization."
        ),
    )

    pdp = "Uniform"
    decay_db = 0.0
    fixed_taps = None

    if tap_model == "Rayleigh fading taps":
        L = st.number_input(
            "Number of Taps (L)",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key=f"{prefix}L",
        )
        pdp = st.selectbox(
            "Power-Delay Profile",
            ["Uniform", "Exponential"],
            key=f"{prefix}pdp",
        )
        if pdp == "Exponential":
            decay_db = st.number_input(
                "Power Decay per Tap (dB)",
                min_value=0.0,
                value=3.0,
                step=1.0,
                key=f"{prefix}decay_db",
            )
        powers = tap_powers(L, pdp, decay_db)
        st.caption(
            "Average tap powers E|h_l|^2: "
            + ", ".join(f"{p:.3f}" for p in powers)
            + " (sum = 1, so the average received symbol energy is A)."
        )
    else:
        tap_text = st.text_input(
            "Tap Coefficients h_0, h_1, ... (comma-separated)",
            value=DEFAULT_FIXED_TAPS,
            key=f"{prefix}fixed_taps",
            help="Complex taps are allowed, e.g. 0.8, 0.5+0.2j, -0.1j.",
        )
        try:
            fixed_taps = parse_taps(tap_text)
        except ValueError as exc:
            st.error(str(exc))
            return
        L = fixed_taps.size
        st.caption(
            "Scaled to unit energy: h = ["
            + ", ".join(format_complex(t) for t in fixed_taps)
            + "]."
        )

    L = int(L)

    # ---------------- equalizers ----------------
    st.subheader("Equalizers")
    equalizers = st.multiselect(
        "Equalizers to Simulate",
        list(EQUALIZERS),
        default=DEFAULT_EQUALIZERS,
        key=f"{prefix}equalizers",
        help=(
            "Every selected equalizer processes the same frames, so their "
            "BERs can be compared directly. Linear/DFE equalizers are "
            "time-domain FIR filters designed for each channel "
            "realization; ZF ignores the noise, MMSE balances ISI and "
            "noise. A DFE also cancels the ISI from its past decisions."
        ),
    )
    if not equalizers:
        st.error("Select at least one equalizer.")
        return

    states = M ** (L - 1)
    trellis_ok = VITERBI not in equalizers or (
        states <= MAX_TRELLIS_STATES and M ** L <= MAX_TRELLIS_BRANCHES
    )
    if VITERBI not in equalizers:
        pass
    elif trellis_ok:
        st.caption(
            f"Viterbi trellis: {states:,} states with {M} branches each."
        )
    else:
        st.error(
            f"The Viterbi trellis would have M^(L-1) = {states:,} states; "
            f"the receiver allows at most {MAX_TRELLIS_STATES:,}. "
            "Lower M or the number of taps, or deselect Viterbi."
        )

    st.divider()

    # ---------------- noise / realizations ----------------
    if sweep:
        st.subheader("SNR Sweep Range (Es/N0)")
        col1, col2, col3 = st.columns(3)
        with col1:
            snr_min = st.number_input(
                "SNR Min (dB)", value=0.0, step=1.0, key=f"{prefix}snr_min",
            )
        with col2:
            snr_max = st.number_input(
                "SNR Max (dB)", value=20.0, step=1.0, key=f"{prefix}snr_max",
            )
        with col3:
            snr_points = st.number_input(
                "Number of Points", min_value=2, value=11, step=1,
                key=f"{prefix}snr_points",
            )

        if snr_max <= snr_min:
            st.error("SNR Max must be greater than SNR Min.")
            return

        if tap_model == "Fixed taps":
            num_repetitions = 1
        else:
            num_repetitions = st.number_input(
                "Channel Realizations per SNR Point",
                min_value=1,
                value=100,
                step=10,
                key=f"{prefix}sweep_repetitions",
                help=(
                    "Each SNR point averages over this many frames, each "
                    "with its own channel taps."
                ),
            )
        st.caption(
            "For each sweep point N0 is computed as A / 10^(SNR_dB / 10) "
            "so that SNR = Es/N0 with Es = A."
        )
    else:
        N0 = st.number_input(
            "Noise Power Spectral Density (N0)",
            min_value=0.0,
            value=0.1,
            step=0.01,
            format="%.6f",
            key=f"{prefix}N0",
        )
        if tap_model == "Fixed taps":
            num_repetitions = 1
        else:
            num_repetitions = st.number_input(
                "Number of Channel Realizations to Average Over",
                min_value=1,
                value=100,
                step=10,
                key=f"{prefix}repetitions",
            )

    if tap_model == "Fixed taps":
        st.caption(
            "Fixed taps: the channel never changes, so no averaging over "
            "channel realizations is needed. The BER comes from one frame "
            "of the message length above; use a longer message for a "
            "more accurate BER."
        )

    st.info(
        "The receiver is given the exact taps of every channel "
        "realization (perfect CSI), and each equalizer is redesigned "
        "for them."
    )

    if st.button("Start Multi-Tap Simulation"):
        k = int(np.log2(M))

        if N % k:
            st.error(
                f"Message length must be divisible by {k} for M={M}."
            )
            return

        if not trellis_ok:
            st.error("Reduce M or the number of taps, or deselect Viterbi.")
            return

        st.session_state[f"{prefix}constellation_type"] = constellation
        st.session_state[f"{prefix}M_value"] = M
        st.session_state[f"{prefix}A_value"] = A
        st.session_state[f"{prefix}N_value"] = int(N)
        st.session_state[f"{prefix}repetitions_value"] = int(num_repetitions)
        st.session_state[f"{prefix}tap_model_value"] = tap_model
        st.session_state[f"{prefix}L_value"] = L
        st.session_state[f"{prefix}pdp_value"] = pdp
        st.session_state[f"{prefix}decay_db_value"] = float(decay_db)
        st.session_state[f"{prefix}fixed_taps_value"] = (
            None if fixed_taps is None else [complex(t) for t in fixed_taps]
        )
        # Keep the order of EQUALIZERS so curves and columns are stable.
        st.session_state[f"{prefix}equalizers_value"] = [
            eq for eq in EQUALIZERS if eq in equalizers
        ]

        if sweep:
            st.session_state["sweep_snr_min"] = float(snr_min)
            st.session_state["sweep_snr_max"] = float(snr_max)
            st.session_state["sweep_num_points"] = int(snr_points)
        else:
            st.session_state[f"{prefix}N0_value"] = float(N0)

        st.session_state[f"{prefix}configured"] = True

        clear_session_keys(result_keys(prefix))
        clear_session_keys([
            f"{prefix}transmitters",
            f"{prefix}receivers",
            f"{prefix}c",
        ])

        st.rerun()


# ============================================================
# RUN CONFIGURED CHANNEL
# ============================================================

def render_configured_simulation(channel):
    prefix = channel_prefix(channel)
    sweep = st.session_state.get("sweep_mode", False)

    initialize_modems(channel)

    button_label = (
        f"Run {channel} BER Sweep" if sweep
        else f"Run {channel} Simulation"
    )

    if st.button(button_label):
        if check_mimo_mode(channel):

            if channel == "AWGN":
                N = st.session_state.N
                A = st.session_state.A
                K = 0.0
                repetitions = (
                    st.session_state.get("awgn_repetitions_value", 1)
                    if sweep else 1
                )
            else:
                N = st.session_state[f"{prefix}N_value"]
                A = st.session_state[f"{prefix}A_value"]
                repetitions = st.session_state[f"{prefix}repetitions_value"]
                K = (
                    st.session_state.rician_K_value
                    if channel == "Rician" else 0.0
                )

            if sweep:
                snr_min = st.session_state.sweep_snr_min
                snr_max = st.session_state.sweep_snr_max
                num_points = int(st.session_state.sweep_num_points)
                snr_list = list(np.linspace(snr_min, snr_max, num_points))

                try:
                    t0 = time.perf_counter()
                    snr_db_list, ber_per_stream, ber_avg = run_sweep(
                        channel=channel,
                        transmitters=st.session_state[f"{prefix}transmitters"],
                        receivers=st.session_state[f"{prefix}receivers"],
                        N=N,
                        A=A,
                        snr_db_list=snr_list,
                        repetitions=repetitions,
                        K=K,
                    )
                    elapsed = time.perf_counter() - t0
                except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
                    st.error(f"Sweep failed: {exc}")
                else:
                    mode = st.session_state.mimo_mode
                    num_results = (
                        st.session_state.numrx
                        if mode == "Transmit Diversity"
                        else st.session_state.numtx
                    )
                    row_label = (
                        "Receiver" if mode == "Transmit Diversity"
                        else "Transmitter"
                    )
                    label_list = [
                        f"{row_label} {i+1}" for i in range(num_results)
                    ]
                    if channel == MULTITAP:
                        label_list = list(
                            st.session_state.multitap_equalizers_value
                        )

                    st.session_state[f"{prefix}sweep_results"] = {
                        "snr_db": snr_db_list,
                        "ber_per_stream": ber_per_stream,
                        "ber_avg": ber_avg,
                        "labels": label_list,
                        "elapsed": elapsed,
                    }
                    st.success(
                        f"{channel} BER sweep completed over "
                        f"{num_points} SNR points!"
                    )
            else:
                N0 = (
                    st.session_state.N0 if channel == "AWGN"
                    else st.session_state[f"{prefix}N0_value"]
                )

                try:
                    result = run_simulation(
                        channel=channel,
                        transmitters=st.session_state[f"{prefix}transmitters"],
                        receivers=st.session_state[f"{prefix}receivers"],
                        N=N,
                        N0=N0,
                        repetitions=repetitions,
                        K=K,
                    )
                except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
                    st.error(f"Simulation failed: {exc}")
                else:
                    store_results(prefix, result)

                    if channel == "AWGN" or is_fixed_multitap(channel):
                        st.success(
                            f"{channel} {st.session_state.mimo_mode} completed!"
                        )
                    else:
                        st.success(
                            f"{channel} {st.session_state.mimo_mode} completed "
                            f"over {repetitions} independent channel "
                            "realizations!"
                        )

    if sweep:
        display_sweep_results(channel)
    else:
        display_simulation_results(channel)


# ============================================================
# DIALOG SELECTION AND SIMULATION EXECUTION
# ============================================================

# IMPORTANT:
# Only ONE dialog function is called per script run.
#
# The new simulation dialog has priority.
# Once confirmed, the selected channel's parameter dialog
# is opened on the next rerun.

if st.session_state.get("open_new_simulation", False):

    simulation_popup()

elif st.session_state.get("channel") == "AWGN":

    if not st.session_state.get("awgn_configured", False):
        awgn_popup()
    else:
        render_configured_simulation("AWGN")

elif st.session_state.get("channel") == "Rician":

    if not st.session_state.get("rician_configured", False):
        rician_popup()
    else:
        render_configured_simulation("Rician")

elif st.session_state.get("channel") == "Rayleigh":

    if not st.session_state.get("rayleigh_configured", False):
        rayleigh_popup()
    else:
        render_configured_simulation("Rayleigh")

elif st.session_state.get("channel") == MULTITAP:

    if not st.session_state.get("multitap_configured", False):
        multitap_popup()
    else:
        render_configured_simulation(MULTITAP)

elif st.session_state.get("channel") == "OFDM":

    st.warning(
        f"{st.session_state.channel} has not been implemented yet."
    )