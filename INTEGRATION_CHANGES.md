# Integration of the Coding Schemes and the General Controller

## Overview

This change does four things:

1. Fixes the bugs that stopped existing code from running.
2. Connects `Transmitter/Coding_Schemes.py` to the transmitter waveform and to the receiver.
3. Adds a headless integration engine that wires **every implemented block** together end to end.
4. Extends the Streamlit controller with a page that exposes every valid combination.

The original controller page and its modes are unchanged. Nothing outside the existing code was implemented: no pilots or channel estimation, no bit or power loading, no adaptation, no Doppler.

```text
streamlit run controllertestv1.py      # GUI (3 pages in the sidebar)
python integration_test.py             # every combination + theory checks (~1.5 min)
python integration_test.py --quick     # noiseless part only (~1 s)
python flow_test.py                    # existing end-to-end test (now runs again)
python Transmitter/chain_test.py       # existing coding-scheme test (runs from any folder)
```

## Files

| File | Status | What |
| --- | --- | --- |
| `flow_test.py` | fixed | wrong import `Channel.LTapChannel` → `Channel.OFDMChannel` |
| `Transmitter/chain_test.py` | fixed | imports work from the project root and from inside `Transmitter/` |
| `Transmitter/__init__.py`, `Channel/__init__.py`, `Receiver/__init__.py` | new (empty) | regular packages, so `Transmitter.Transmitter` always resolves the same way |
| `Transmitter/Transmitter.py` | refactor, same outputs | new `Transmitter.symbols()` / `map_bits()`; `run()` and `modulate()` unchanged |
| `Transmitter/Coding_Schemes.py` | additions only | `SpaceTimeTransmitter`, `alamouti_ofdm_map/unmap`, `DifferentialPSK`, `DifferentialAlamouti` |
| `Receiver/receiver.py` | 2 bug fixes + additions | Alamouti bias fix, `selection_combine` fix; `repetition_combine`, `decode_repetition`, `alamouti_ofdm_combine`, `decode_alamouti_ofdm` |
| `Integration/link_engine.py` | new | general link engine (no Streamlit) |
| `pages/1_General_Link_Simulator.py` | new | GUI: every combination |
| `pages/2_Integration_Self_Test.py` | new | GUI: self-test of every combination + `monte_carlo` vs `theory_ser_qam` |
| `controllertestv1.py` | 2 small edits | OFDM placeholder now opens the General Link Simulator; one caption |
| `integration_test.py` | new | command-line verification |
| `requirements.txt` | new | dependencies |

The channel modules (`Channel/*.py`) were **not modified**.

## 1. Bug fixes

| Problem | Effect | Fix |
| --- | --- | --- |
| `flow_test.py` imported `Channel.LTapChannel`, which does not exist | the script crashed at import | import `LTapChannel` from `Channel/OFDMChannel.py` |
| `chain_test.py` used `from Transmitter import ...` | only ran from inside `Transmitter/` | inserts the project root in `sys.path` and adds package `__init__.py` files |
| `Receiver.alamouti_combine` divided by g + N0/Es when N0 was given (report Sec. 3.5) | shrunk amplitudes and raised the 16-QAM BER | always the unbiased z/g; `N0` is still accepted for compatibility |
| `Receiver.selection_combine` with a constant H of shape (Nr,) | `IndexError` (it indexed H per column) | H is broadcast to (Nr, T) first |
| Controller "OFDM" entry | "has not been implemented yet" | points to the General Link Simulator, where OFDM works with every scheme |

## 2. Connection of the coding schemes with the other blocks

### Transmit chain (code first, then waveform per antenna)

```text
bits --Transmitter.symbols--> s (Ns,) --CodingScheme.encode--> X (Nt, T)
     --[Alamouti + OFDM: alamouti_ofdm_map]--> Waveform.modulate(row) per antenna
     --> samples (Nt, n_samples) = X of Y = H X + N
```

* `Transmitter.symbols(bits, N)` runs the same checks as `run()` but returns the constellation symbols *before* the waveform. A coding scheme must act on symbols. With the old code, `run()` returned OFDM time samples, and Alamouti or V-BLAST would have been applied to time samples.
* `SpaceTimeTransmitter(transmitter, scheme)` performs the chain above. Its `run()` output can be passed directly to `AWGN_Channel`, `MIMORicianFlatChannel` or `MIMOLTapChannel`.
* **Alamouti over OFDM.** Adjacent subcarriers have different gains over multipath, but the same subcarrier in two consecutive OFDM blocks has the same gain when the channel is block-static. So pair m goes to subcarrier m mod n of blocks 2(m div n) and 2(m div n)+1. On every subcarrier the receiver then runs the ordinary Alamouti combiner with that subcarrier's 2-antenna channel. This is exact, not an approximation. The frame needs an even number of OFDM blocks.
* `DifferentialPSK` and `DifferentialAlamouti` put the encoders that were written with their detectors (`receiver.differential_encode`, `receiver.differential_alamouti_encode`) behind the same `CodingScheme` interface. The encoder code itself stays in `receiver.py`.

### Receive side (matching methods, same `power_split`)

| Scheme | Flat channel (AWGN, block fading) | Multipath |
| --- | --- | --- |
| V-BLAST / SISO | `decode_mimo` (zf, mmse, sic), `decode_mlse_mimo` (joint ML), `decode_mrc/egc/selection` (Nt = 1) | SC: `decode_sc` (linear/DFE, ZF/MMSE), `decode_mlse_mimo` (Viterbi, M-algorithm, K-best, shortening); SC+CP / OFDM: `decode_ofdm_mimo` |
| Repetition(L) | **new** `decode_repetition` (MRC / EGC / selection over Nr·L branches) | OFDM: `decode_repetition` with per-subcarrier gains; SC / SC+CP: equalize, then average the L copies |
| Alamouti | `decode_alamouti` (SC), **new** `decode_alamouti_ofdm` (OFDM) | OFDM: `decode_alamouti_ofdm` |
| DifferentialPSK / DifferentialAlamouti | `decode_differential` / `decode_differential_alamouti` | — |

## 3. General link engine (`Integration/link_engine.py`)

`LinkConfig` holds every choice and `Link(cfg)` builds the chain. `Link.simulate()` and `Link.sweep()` run Monte Carlo frames. All selected detectors process the *same* bits, channel and noise, so their results are directly comparable.

`detectors_for`, `channels_for`, `waveforms_for` and `validate` define which combinations exist. The GUI only offers those. `chain_description(cfg)` lists the exact module function called in every block, and the GUI shows it in a "Signal chain" box.

### Conventions (identical for every combination)

* Y = H X + N, with X of shape (Nt, channel uses) and noise CN(0, N0) per complex receive sample.
* With `power_split=True` (default, the transmitter convention), the total energy per channel use is A for every Nt. With `False`, each antenna sends A. The receiver is always called with the same setting.
* SNR is either **Es/N0** with N0 = A / SNR (the original controller's convention), or **Eb/N0**, where Eb is the transmitted energy per information bit. Eb includes the cyclic-prefix, reference-symbol and repetition overhead and the sum over all transmit antennas.
* α-stable noise: the channel is called with Nzero = 0 and S-α-S noise is added. For detectors that need a noise level, N0 = 4γ² (the α = 2 equivalence, checked numerically). A sweep uses γ = √(N0/4).
* AWGN with several antennas uses a fixed unit-modulus DFT-phase H (the original controller's `los_matrix`). It is full rank for Nr ≥ Nt.
* Perfect CSI: the receiver gets the channel realization returned by the channel function.

### Supported combinations

| Scheme (Nt) | Waveforms | Channels |
| --- | --- | --- |
| SISO / SIMO (1) | SC, SC+CP, OFDM | AWGN, fast Rayleigh/Rician, block Rayleigh/Rician, multipath, legacy OFDM link (SC) |
| Repetition L (1) | SC, SC+CP, OFDM | AWGN, fast (SC, SC+CP), block, multipath, legacy OFDM link (SC) |
| Alamouti (2) | SC, SC+CP, OFDM | AWGN, block; multipath with OFDM |
| V-BLAST (≥2) | SC, SC+CP, OFDM | AWGN, block, multipath |
| Differential PSK (1) | SC | AWGN, fast, block |
| Differential Alamouti (2) | SC | AWGN, block |
| Orthogonal M-FSK (1) | correlator domain | AWGN, block |

Constellations: PAM, square QAM, PSK, unipolar ASK/OOK, and any custom mapping (`Constellation(..., mapping=dict)`). ASK and custom mappings with distinct magnitudes enable energy detection.

Noise: Gaussian on every channel; α-stable on every channel except the legacy OFDM link, which adds its own Gaussian noise internally.

Antennas: any Nr. V-BLAST supports Nt from 2 to 8.

Detectors are filtered by the constraints the receiver enforces, for example:
* ZF needs Nr ≥ Nt.
* Full Viterbi is only offered when the trellis fits the receiver's limits.
* Exhaustive ML requires M^Nt ≤ 4096.

## 4. GUI

* **Classic simulator** (`controllertestv1.py`): the original AWGN / Rayleigh / Rician / multi-tap modes, unchanged.
* **General Link Simulator**: transmitter, channel and receiver panels. Results are available as a single point or a BER/SER sweep:
  * several detectors compared at once
  * Es/N0, Eb/N0 or N0 input
  * BER, SER, error counts, rate and goodput
  * a bit table
  * the constellation with its Gray labels
  * the last channel realization
  * the theory overlay (`receiver.theory_ser_qam`, AWGN SISO QAM)
  * CSV export
  * a fixed seed for reproducibility
* **Integration Self-Test**: noiseless run of all 184 configurations (636 configuration × detector checks), and `receiver.monte_carlo` versus `theory_ser_qam`.

## 5. Verification

* **Noiseless:** 184 configurations and 636 detector runs gave 584 PASS (exact detectors, zero errors), 52 INFO and 0 FAIL. INFO marks cases where some errors are expected by design: finite-length linear/DFE equalizers and the M-algorithm (these were also error-free here), differential detection under per-sample fading, and energy detection under fading.
* **Theory:** 18 Monte Carlo checks, all within 20%, and most within 5%:
  * 16-QAM SER for SC, SC+CP and OFDM
  * BPSK-OFDM at Eb/N0, with the CP energy charged
  * 1×1, 1×2 and 1×3 MRC, and repetition L = 3, against Rayleigh MRC theory
  * Alamouti 2×1 and 2×2 (including over OFDM and over multipath OFDM)
  * V-BLAST 2×2 and 2×3 ZF (diversity Nr − Nt + 1)
  * per-subcarrier MMSE over multipath, and the legacy OFDM link
  * α-stable noise with α = 2 against AWGN at N0 = 4γ²
  * binary DPSK and non-coherent FSK
* **Regression:** `Transmitter.run` and `modulate` give bit-identical outputs and identical errors compared with the previous version (88 cases).
* **GUI:** all 69 scheme × waveform × channel paths, sweeps, α-stable noise, custom and ASK constellations, both new pages, and the original Rayleigh and OFDM flows were run with Streamlit's AppTest without an exception.

## 6. Not done on purpose (pending in the progress report)

* Pilot insertion, frame structure and pilot-based or imperfect-CSI estimation. `pilot_matrix` and `estimate_channel` exist but are not connected.
* Bit or power loading, the adaptation controller, soft (LLR) outputs.
* Doppler or time-correlated fading, and SISO block fading.

## 7. Notes for the other teams (code not changed)

* `Alpha_Stable_Channel` draws `Data.size` noise samples as a **1-D** array. It therefore fails, or broadcasts wrongly, for 2-D (Nr, T) data. The engine flattens and reshapes around it.
* `MIMOLTapChannel` and `OFDMChannel` use pure-Python double-loop convolutions and DFTs. Replacing them with `np.convolve` and `np.fft` would give identical results, much faster.
* The classic page's spatial multiplexing uses `power_split=False` (energy A per antenna), and its AWGN mode uses H = H_los/√Nr. Both are kept as they were. The General Link Simulator uses the transmitter convention by default and lets you choose either power convention.
