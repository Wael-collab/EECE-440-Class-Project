# Waveform Support and Flow-Test Changes

## Overview

The chain is now `bits -> constellation symbols -> waveform samples -> channel -> waveform demodulation -> symbol detection -> bits`. A shared waveform configuration selects single-carrier (SC) or OFDM and whether a cyclic prefix (CP) is used. The flow test compares OFDM with a CP against SC without one across the available channel models.

## Files changed

- [`Transmitter/Transmitter.py`](./Transmitter/Transmitter.py)
- [`Receiver/receiver.py`](./Receiver/receiver.py)
- [`flow_test.py`](./flow_test.py)

## New `Waveform` class

`Transmitter/Transmitter.py` now contains a `Waveform` class with two supported modes:

- `"SC"`: single-carrier transmission at one sample per constellation symbol. With a cyclic prefix, symbols are grouped into blocks of `n_subcarriers` and each block receives its own prefix.
- `"OFDM"`: frequency-domain symbols become time-domain samples through a unitary IFFT; the receiver uses the matching unitary FFT.

The class accepts the following configuration:

```python
Waveform("SC")
Waveform("SC", n_subcarriers=64, cyclic_prefix=16)
Waveform("OFDM", n_subcarriers=64)
Waveform("OFDM", n_subcarriers=64, cyclic_prefix=16)
```

The aliases `"SINGLE_CARRIER"` and `"SINGLE-CARRIER"` are also accepted for single-carrier operation.

`cyclic_prefix` must be a non-negative integer; the default `0` disables it. OFDM always needs a positive integer `n_subcarriers`. SC needs one only when the CP is nonzero, in which case `n_subcarriers` means **SC block size**. If SC has no CP, `n_subcarriers` is ignored. For either waveform, a nonzero CP must be shorter than the block size. Invalid settings raise `ValueError`.

### Single-carrier cyclic prefix

With a CP, the transmitter divides the symbols into complete blocks of `n_subcarriers`, prepends the last `cyclic_prefix` symbols **to each block**, then concatenates the blocks. The receiver splits the samples into blocks of `n_subcarriers + cyclic_prefix` and removes each prefix. For example, 128 symbols with a block size of 64 and a CP of 16 produce 160 samples. Without a CP, SC simply copies the one-sample-per-symbol stream; no block size is needed. SC here is symbol-rate pass-through, not pulse shaping or upsampling.

### OFDM processing

For OFDM, the input symbols are divided into blocks of `n_subcarriers` symbols. Each block is processed as follows:

1. Apply a unitary IFFT to convert frequency-domain symbols to time-domain samples.
2. Copy the last `cyclic_prefix` samples to the beginning of each block.
3. Flatten the blocks into a one-dimensional waveform for transmission.

At the receiver:

1. Split the received waveform into complete OFDM blocks.
2. Remove the cyclic prefix from every block.
3. Apply the matching unitary FFT.
4. Return the recovered constellation-symbol stream.

For OFDM, a block must contain exactly `n_subcarriers` symbols. Both OFDM and SC-with-CP reject incomplete transmit or receive blocks rather than silently padding them. The unitary FFT/IFFT preserves the energy of the OFDM data portion; copying CP samples adds transmitted energy.

## Transmitter changes

`Transmitter` now accepts an optional waveform:

```python
tx = Transmitter(constellation, waveform)
```

If no waveform is supplied, the transmitter defaults to single-carrier operation for backward compatibility:

```python
tx = Transmitter(constellation)
```

The bit-to-symbol mapping process is unchanged. After mapping the bits to constellation symbols, `Transmitter.run()` passes those symbols to `waveform.modulate()` and returns **samples**, not constellation symbols. For `B` complete blocks of `n_subcarriers` symbols, the output has `B * (n_subcarriers + cyclic_prefix)` samples. CP-free SC returns one sample per symbol.

For OFDM and single-carrier with a cyclic prefix, the number of generated constellation symbols must be divisible by `n_subcarriers`. Otherwise, `Waveform.modulate()` raises a `ValueError` instead of silently padding the transmission.

## Receiver changes

`Receiver` now accepts the same waveform configuration as the transmitter:

```python
rx = Receiver(constellation, waveform)
```

It also defaults to single-carrier operation when no waveform is supplied:

```python
rx = Receiver(constellation)
```

Before minimum-distance constellation detection, `Receiver.detect()` calls `waveform.demodulate()`. This removes any single-carrier or OFDM cyclic prefix and, for OFDM, applies the FFT. `Receiver.decode()` therefore accepts channel output samples directly for either waveform.

The receiver and transmitter must use matching settings: mode, block size when applicable, and CP length. `Receiver.decode(samples, h)` applies `h` **after** waveform demodulation, so `h` must be a scalar or a gain array aligned with the recovered constellation symbols. Sample-domain fading gains for OFDM must instead be removed before calling `decode`.

## End-to-end flow test

Run [`flow_test.py`](./flow_test.py) from the project root. It compares the following two configurations using the **same 4096 input bits** and 16-QAM (`M=16`, `A=4`):

```python
for name, waveform in (
    ("OFDM + CP", Waveform("OFDM", n_subcarriers=64, cyclic_prefix=16)),
    ("SC without CP", Waveform("SC")),
):
    transmitter = Transmitter(constellation, waveform)
    receiver = Receiver(constellation, waveform)
    transmitted = transmitter.run(bits, len(bits))
    # Decode directly for the noiseless test, then run each channel model.
```

Each 16-QAM symbol represents four bits, so 4096 bits produce 1024 symbols. CP-free SC sends **1024 samples**. OFDM uses sixteen 64-subcarrier blocks; adding 16 CP samples to each produces **1280 samples**. Both noiseless round trips must reproduce every input bit or the script fails an assertion.

The script prints BER, SER and bit-error counts for each of these cases:

| Channel case | SC without CP | OFDM with CP |
| --- | --- | --- |
| Noiseless | Direct symbol detection | Remove CP, FFT, then detect |
| AWGN | Decode noisy samples | Remove CP, FFT, then detect |
| Rayleigh | Divide by the returned sample gains in `Receiver.decode` | Divide the received **time-domain samples** by their returned gains before the FFT |
| Rician (`K=1`) | Same coherent sample-gain correction as Rayleigh | Same time-domain correction as Rayleigh |
| Four-tap channel | Decode without an ISI equalizer; labeled `unequalized ISI` | Trim the convolution tail, remove the CP, and equalize each subcarrier with the FFT of the channel taps |
| Alpha-stable (`alpha=1.8`, `gamma=0.1`) | Decode noisy samples if `aub_htp` is installed | Decode noisy samples if `aub_htp` is installed |

The AWGN, Rayleigh, Rician and L-tap cases use nominal `Eb/N0 = 10 dB` and `N0 = (A / bits_per_symbol) / 10^(Eb/N0_dB / 10)`. This calculation does **not** charge the extra CP energy against the bit-energy budget. The alpha-stable channel uses its separate `gamma` scale, not `N0`, so its BER is not an equal-`Eb/N0` comparison. The channel functions use NumPy's global random generator; the script seeds it and also seeds the input-bit generator for repeatable runs.

The Rayleigh and Rician implementations draw an independent gain for **every sample**, including OFDM CP samples. Their OFDM check therefore equalizes in the time domain; a single gain per subcarrier would not describe those channels. The SC L-tap result includes intersymbol interference because CP-free SC has no equalizer in this test. Its BER should not be read as a coherently equalized SC-versus-OFDM comparison.

[`Channel/OFDMChannel.py`](./Channel/OFDMChannel.py) is tested **separately** on one 64-symbol input block. That legacy function already performs an IDFT, adds a CP, applies an L-tap channel, removes the CP and performs a DFT. Passing samples from the new OFDM waveform into it would apply OFDM twice. The standalone result uses the FFT of the returned channel taps for coherent subcarrier equalization.

The optional plot shows the ideal 16-QAM points and the **demodulated OFDM + AWGN** symbols:

```python
ofdm_awgn_symbols = waveform.demodulate(noisy)
```

If Matplotlib is unavailable, the simulation still runs and prints `Plot skipped`. If `aub_htp` is unavailable, each alpha-stable result is printed as skipped; the other channels still run.

## Additional cleanup

The demonstration code at the bottom of `Transmitter.py` is now protected by:

```python
if __name__ == "__main__":
```

This prevents random test output from being printed whenever the transmitter module is imported by the receiver or another script.

## Validation completed

- Both configurations in `flow_test.py` recover all 4096 input bits without noise (BER and SER both zero).
- The full flow test exits successfully without Matplotlib or `aub_htp`; those optional outputs are explicitly skipped.
- A separate noiseless OFDM plus four-tap-channel check recovered every bit after subcarrier equalization.
- A separate multi-block SC-with-CP check verified prefix placement and bit recovery; CP-free SC with an irrelevant block-size argument also recovered every bit.
- VS Code diagnostics found no errors in the modified Python files during validation.
