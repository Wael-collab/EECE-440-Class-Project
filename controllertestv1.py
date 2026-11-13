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


def display_bit_comparison(bits, decoded_bits, num_rows, N, row_label="Transmitter"):
    """Display sent/decoded bits; rows can be transmitters or receivers."""
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
            f"{row_label} {i+1}</td>"
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
    """
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
    prefix = "" if channel == "AWGN" else f"{channel.lower()}_"
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
    prefix = "" if channel == "AWGN" else f"{channel.lower()}_"
    state = st.session_state

    if f"{prefix}received" not in state:
        return

    mode = state.mimo_mode
    numtx = state.numtx
    numrx = state.numrx
    N = state.N if channel == "AWGN" else state[f"{prefix}N_value"]

    labels = "Receiver" if mode == "Transmit Diversity" else "Transmitter"
    num_results = numrx if mode == "Transmit Diversity" else numtx

    st.subheader(f"{channel} Simulation Results")

    if channel == "Rician":
        st.write(f"Rician K-factor (linear): {state.rician_K_value}")
    elif channel == "Rayleigh":
        st.write("Channel model: Rayleigh fading (K = 0)")

    if channel != "AWGN":
        st.write("Detection mode: Coherent (Perfect CSI)")
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

    timing_key = "simulation_timings" if not prefix else f"{prefix}timings"

    display_timing_breakdown(
        state[timing_key],
        "Computational time only; browser rendering is excluded. "
        + (
            "Timings are accumulated across all channel realizations."
            if channel != "AWGN" else ""
        ),
    )

    if channel != "AWGN":
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
    )

    st.subheader(
        "Monte Carlo BER (Averaged over channel realizations)"
        if channel != "AWGN" else "Bit Error Rates"
    )

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
                f"**{labels} {i+1}:** "
                f"{state[f'{prefix}error_counts'][i]:,} errors out of "
                f"{state[f'{prefix}total_bits'][i]:,} bits"
            )


def display_sweep_results(channel):
    """Render the BER-vs-SNR sweep curve and raw data."""
    prefix = "" if channel == "AWGN" else f"{channel.lower()}_"
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
    if len(ber_per_stream) > 1:
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
    else:
        mimo_mode = st.selectbox(
            "MIMO Transmission Mode",
            MODES,
            index=2,
            key="initial_mimo_mode",
        )

    # In sweep mode, diversity modes collapse to a single BER curve.
    if sweep and mimo_mode == "Receive Diversity":
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


# ============================================================
# RUN CONFIGURED CHANNEL
# ============================================================

def render_configured_simulation(channel):
    prefix = "" if channel == "AWGN" else f"{channel.lower()}_"
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

                    if channel == "AWGN":
                        st.success(
                            f"AWGN {st.session_state.mimo_mode} completed!"
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

elif st.session_state.get("channel") in ("OFDM", "Multi-Tap Channel"):

    st.warning(
        f"{st.session_state.channel} has not been implemented yet."
    )