import numpy as np

from Transmitter import Constellation, Transmitter
from Coding_Schemes import Repetition, Alamouti, VBLAST

rng = np.random.default_rng(7)
N = 4000
bits = rng.integers(0, 2, N)

constellation = Constellation("QAM", M=16, A=4)
tx = Transmitter(constellation)
symbols = tx.run(bits, N)
Es = np.mean(np.abs(symbols) ** 2)
print("symbols:", symbols.shape)

# Common contract: every scheme returns (n_tx, channel_uses) with total energy per use = Es
for scheme in [Repetition(3), Alamouti(), VBLAST(2), VBLAST(4)]:
    X = scheme.encode(symbols)
    assert X.shape == (scheme.n_tx, scheme.channel_uses(len(symbols)))
    assert np.isclose(np.mean(np.sum(np.abs(X) ** 2, axis=0)), Es)
    print(f"{type(scheme).__name__:10s} n_tx={scheme.n_tx} shape={X.shape} "
          f"channel_uses={scheme.channel_uses(len(symbols))}")

# Repetition ordering
rep = Repetition(3).encode(symbols)
assert np.allclose(rep[0].reshape(-1, 3), symbols[:, None])

# Alamouti: noiseless combining check, one (h1, h2) per symbol pair
X = Alamouti(power_split=False).encode(symbols)
h1, h2 = (rng.standard_normal((2, len(symbols) // 2))
          + 1j * rng.standard_normal((2, len(symbols) // 2))) / np.sqrt(2)
y1 = h1 * X[0, 0::2] + h2 * X[1, 0::2]
y2 = h1 * X[0, 1::2] + h2 * X[1, 1::2]
g = np.abs(h1) ** 2 + np.abs(h2) ** 2
u1 = (np.conj(h1) * y1 + h2 * np.conj(y2)) / g
u2 = (np.conj(h2) * y1 - h1 * np.conj(y2)) / g
assert np.allclose(u1, symbols[0::2]) and np.allclose(u2, symbols[1::2])

# V-BLAST layout
for n_tx in (2, 4):
    V = VBLAST(n_tx, power_split=False).encode(symbols)
    assert np.allclose(V.T.reshape(-1), symbols)

# Invalid lengths
for bad in [lambda: Alamouti().encode(symbols[:5]), lambda: VBLAST(3).encode(symbols)]:
    try:
        bad()
        raise SystemExit("expected ValueError")
    except ValueError as e:
        print("ValueError -", e)

print("all checks passed")