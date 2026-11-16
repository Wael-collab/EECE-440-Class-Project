"""
End-to-end integration test for the whole class system (run from the project root):

    python integration_test.py          # full test (about a minute)
    python integration_test.py --quick  # noiseless part only

Part 1 - noiseless round trip of EVERY supported combination of
         scheme x waveform x channel x antennas x detector (Integration.link_engine).
Part 2 - Monte Carlo versus closed-form theory, one check per type of connection, so
         that the SNR / power-split / cyclic-prefix / repetition conventions are verified
         and not only the array shapes.
"""

import math
import sys
import time

import numpy as np

from Integration.link_engine import LinkConfig, Link, self_test, n0_from_snr
from Receiver.receiver import qfunc, theory_ser_qam


def rayleigh_mrc_bpsk(g, L):
    """BPSK with L-branch MRC over i.i.d. Rayleigh fading, average SNR g per branch."""
    mu = math.sqrt(g / (1 + g))
    return ((1 - mu) / 2) ** L * sum(
        math.comb(L - 1 + l, l) * ((1 + mu) / 2) ** l for l in range(L))


def check(name, sim, theory, tol=0.2):
    ok = abs(sim - theory) <= tol * theory
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<62} sim={sim:.3e}  theory={theory:.3e}")
    return ok


def run_point(cfg, snr_db, n_bits, frames, definition="EsN0", seed=1, det=None):
    link = Link(cfg)
    N0 = n0_from_snr(cfg, snr_db, n_bits, definition)
    res = link.simulate(n_bits, N0, frames, seed=seed)["per_detector"]
    return res[det or cfg.detectors[0]]


def statistical_checks():
    ok = True
    g = lambda db: 10 ** (db / 10)

    print("\nAWGN, square QAM (receiver.theory_ser_qam), every waveform:")
    for wf in ("SC", "SC_CP", "OFDM"):
        cfg = LinkConfig(constellation="QAM", M=16, A=4.0, waveform=wf, n_subcarriers=64,
                         cyclic_prefix=16, channel="AWGN", detectors=["ZF"])
        r = run_point(cfg, 14, 64 * 4 * 16, 20)
        ok &= check(f"16-QAM SER, {wf}, Es/N0 = 14 dB", r["ser"], theory_ser_qam(16, g(14)))

    print("\nEb/N0 bookkeeping (CP energy charged to the bits):")
    cfg = LinkConfig(constellation="PAM", M=2, waveform="OFDM", n_subcarriers=64,
                     cyclic_prefix=16, channel="AWGN", detectors=["ZF"])
    r = run_point(cfg, 7, 64 * 64, 25, definition="EbN0")
    ok &= check("BPSK-OFDM BER at Eb/N0 = 7 dB: Q(sqrt(2 g 64/80))", r["ber"],
                qfunc(math.sqrt(2 * g(7) * 64 / 80)))

    print("\nFast Rayleigh fading, receive diversity and repetition (MRC theory):")
    for nr in (1, 2, 3):
        cfg = LinkConfig(constellation="PAM", M=2, channel="RAYLEIGH_FAST", n_rx=nr,
                         detectors=["MRC"])
        r = run_point(cfg, 10, 20000, 10)
        ok &= check(f"BPSK, 1x{nr} MRC, Es/N0 = 10 dB", r["ber"], rayleigh_mrc_bpsk(g(10), nr))
    cfg = LinkConfig(constellation="PAM", M=2, scheme="REPETITION", rep_L=3,
                     channel="RAYLEIGH_FAST", detectors=["MRC"])
    r = run_point(cfg, 6, 20000, 10)
    ok &= check("BPSK, Repetition L=3 over fast fading (= 3-branch MRC)", r["ber"],
                rayleigh_mrc_bpsk(g(6), 3))

    # Block fading: all bits of a frame share one H, so the number of independent
    # channel realizations (frames), not the number of bits, sets the accuracy.
    print("\nFlat block fading, space-time codes (power split 1/sqrt(Nt)):")
    cfg = LinkConfig(constellation="PAM", M=2, scheme="ALAMOUTI", n_rx=1,
                     channel="RAYLEIGH_BLOCK", detectors=["ALAMOUTI"])
    r = run_point(cfg, 12, 32, 30000)
    ok &= check("BPSK Alamouti 2x1 (= 2-branch MRC at g/2)", r["ber"],
                rayleigh_mrc_bpsk(g(12) / 2, 2))
    cfg = LinkConfig(constellation="PAM", M=2, scheme="ALAMOUTI", n_rx=2,
                     channel="RAYLEIGH_BLOCK", waveform="OFDM", n_subcarriers=16,
                     cyclic_prefix=4, detectors=["ALAMOUTI"])
    r = run_point(cfg, 8, 32, 40000)
    ok &= check("BPSK Alamouti 2x2 over OFDM (= 4-branch MRC at g/2)", r["ber"],
                rayleigh_mrc_bpsk(g(8) / 2, 4))
    cfg = LinkConfig(constellation="PAM", M=2, scheme="VBLAST", n_tx=2, n_rx=2,
                     channel="RAYLEIGH_BLOCK", detectors=["ZF"])
    r = run_point(cfg, 15, 32, 30000)
    ok &= check("BPSK V-BLAST 2x2 ZF (diversity Nr-Nt+1 = 1 at g/2)", r["ber"],
                rayleigh_mrc_bpsk(g(15) / 2, 1))
    cfg.n_rx = 3
    r = run_point(cfg, 10, 32, 30000)
    ok &= check("BPSK V-BLAST 2x3 ZF (diversity 2 at g/2)", r["ber"],
                rayleigh_mrc_bpsk(g(10) / 2, 2))

    print("\nMultipath + OFDM: each subcarrier is flat Rayleigh (uniform PDP, power 1):")
    for name, cfg in (
        ("SISO OFDM, per-subcarrier MMSE", LinkConfig(
            constellation="PAM", M=2, channel="MULTIPATH", L_taps=4, waveform="OFDM",
            n_subcarriers=64, cyclic_prefix=8, detectors=["OFDM_MMSE"])),
        ("Alamouti 2x1 over multipath OFDM (= 2-branch MRC at g/2)", LinkConfig(
            constellation="PAM", M=2, scheme="ALAMOUTI", channel="MULTIPATH", L_taps=4,
            waveform="OFDM", n_subcarriers=64, cyclic_prefix=8, detectors=["ALAMOUTI"])),
    ):
        r = run_point(cfg, 12, 128 * 2, 300)
        L = 2 if cfg.scheme == "ALAMOUTI" else 1
        ok &= check(name, r["ber"], rayleigh_mrc_bpsk(g(12) / L, L))

    print("\nLegacy self-contained OFDM link (Channel/OFDMChannel.OFDM):")
    cfg = LinkConfig(constellation="PAM", M=2, channel="LEGACY_OFDM", L_taps=4,
                     legacy_block=64, detectors=["ONE_TAP_ZF"])
    r = run_point(cfg, 12, 64 * 4, 200)
    ok &= check("BPSK one-tap ZF (= flat Rayleigh per subcarrier)", r["ber"],
                rayleigh_mrc_bpsk(g(12), 1))

    from Integration.link_engine import ALPHA_STABLE_AVAILABLE
    if ALPHA_STABLE_AVAILABLE:
        print("\nalpha-stable noise (Channel/AWSN.py), alpha = 2 must equal AWGN with N0 = 4 gamma^2:")
        cfg = LinkConfig(constellation="QAM", M=16, A=1.0, channel="AWGN", noise="ALPHA_STABLE",
                         alpha=2.0, detectors=["ZF"])
        link, gamma = Link(cfg), 0.07
        r = link.simulate(64 * 4 * 16, 4 * gamma ** 2, 100, gamma=gamma, seed=3)["per_detector"]["ZF"]
        ok &= check("16-QAM SER with SaS(alpha=2, gamma=0.07)", r["ser"],
                    theory_ser_qam(16, 1.0 / (4 * gamma ** 2)))
    else:
        print("\n(alpha-stable check skipped: aub_htp is not installed)")

    print("\nNon-coherent detectors over AWGN:")
    cfg = LinkConfig(constellation="PSK", M=2, scheme="DPSK", channel="AWGN",
                     detectors=["DIFF"])
    r = run_point(cfg, 7, 20000, 10)
    ok &= check("Binary DPSK: 0.5 exp(-Eb/N0)", r["ber"], 0.5 * math.exp(-g(7)))
    cfg = LinkConfig(constellation="PAM", M=2, scheme="FSK", channel="AWGN",
                     detectors=["FSK"])
    r = run_point(cfg, 11, 20000, 10)
    ok &= check("Binary orthogonal FSK: 0.5 exp(-Eb/(2 N0))", r["ber"], 0.5 * math.exp(-g(11) / 2))
    return ok


def noiseless_checks():
    rows = self_test()
    fails = [r for r in rows if r["status"].startswith("FAIL")]
    n_pass = sum(r["status"] == "PASS" for r in rows)
    n_info = sum(r["status"] == "INFO" for r in rows)
    print(f"Noiseless round trip: {n_pass} PASS, {n_info} INFO (approximate detectors), "
          f"{len(fails)} FAIL over {len({r['combo'] for r in rows})} configurations.")
    for r in fails:
        print("  FAIL:", r["combo"], r["detector"], r["status"])
    return not fails


if __name__ == "__main__":
    t0 = time.time()
    ok = noiseless_checks()
    if "--quick" not in sys.argv:
        ok &= statistical_checks()
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} ({time.time() - t0:.1f} s)")
    sys.exit(0 if ok else 1)
