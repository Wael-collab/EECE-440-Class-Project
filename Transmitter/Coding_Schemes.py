import numpy as np


class CodingScheme:
    """
    Base class for transmit-side coding schemes. Input is the output of
    Transmitter.run (a 1-D complex array of constellation symbols); output
    of encode() is a complex array of shape (n_tx, T): rows = antennas,
    columns = channel uses, i.e. the X in Y = HX + N.

    n_tx        : number of transmit antennas
    power_split : if True, each antenna is scaled by 1/sqrt(n_tx) so the
                  total energy per channel use stays A.
    """

    def __init__(self, n_tx, power_split=True):
        if not isinstance(n_tx, (int, np.integer)) or n_tx <= 0:
            raise ValueError("n_tx must be a positive integer.")
        self.n_tx = int(n_tx)
        self.power_split = power_split

    @staticmethod
    def _check_symbols(symbols):
        symbols = np.asarray(symbols)
        if symbols.ndim != 1 or symbols.size == 0:
            raise ValueError("symbols must be a nonempty 1-D array.")
        if not np.all(np.isfinite(symbols)):
            raise ValueError("symbols must contain finite values.")
        return symbols.astype(complex)

    def _scale(self, X):
        return X / np.sqrt(self.n_tx) if self.power_split else X

    def encode(self, symbols):
        raise NotImplementedError

    def channel_uses(self, n_symbols):
        """Number of channel uses (columns of X) needed for n_symbols symbols."""
        raise NotImplementedError


class Repetition(CodingScheme):
    """
    Repeats each symbol L times (x_l = x for l = 1..L) on a single antenna.
    Columns are ordered [s1 s1 ... s1, s2 s2 ... s2, ...]; the receiver can
    reshape(-1, L).
    """

    def __init__(self, L):
        if not isinstance(L, (int, np.integer)) or L <= 0:
            raise ValueError("L must be a positive integer.")
        super().__init__(n_tx=1)
        self.L = int(L)

    def encode(self, symbols):
        s = self._check_symbols(symbols)
        return self._scale(np.repeat(s, self.L)[None, :])

    def channel_uses(self, n_symbols):
        return n_symbols * self.L


class Alamouti(CodingScheme):
    """
    2-transmit-antenna Alamouti code. Symbols are taken in pairs (u1, u2):

        antenna 1:  u1   -u2*
        antenna 2:  u2    u1*      (columns = channel uses)

    Needs an even number of symbols.
    """

    def __init__(self, power_split=True):
        super().__init__(n_tx=2, power_split=power_split)

    def encode(self, symbols):
        s = self._check_symbols(symbols)
        self.channel_uses(len(s))

        s1, s2 = s[0::2], s[1::2]
        X = np.zeros((2, len(s)), dtype=complex)
        X[0, 0::2] = s1
        X[0, 1::2] = -np.conj(s2)
        X[1, 0::2] = s2
        X[1, 1::2] = np.conj(s1)

        return self._scale(X)

    def channel_uses(self, n_symbols):
        if n_symbols % 2 != 0:
            raise ValueError("Alamouti coding needs an even number of symbols.")
        return n_symbols


class VBLAST(CodingScheme):
    """
    V-BLAST spatial multiplexing: the symbol stream is split across n_tx
    antennas, so each channel use carries n_tx independent symbols.
    Needs a number of symbols that is a multiple of n_tx.
    """

    def __init__(self, n_tx, power_split=True):
        super().__init__(n_tx=n_tx, power_split=power_split)

    def encode(self, symbols):
        s = self._check_symbols(symbols)
        self.channel_uses(len(s))

        return self._scale(s.reshape(-1, self.n_tx).T.copy())

    def channel_uses(self, n_symbols):
        if n_symbols % self.n_tx != 0:
            raise ValueError(
                f"Number of symbols ({n_symbols}) must be a multiple of n_tx = {self.n_tx}."
            )
        return n_symbols // self.n_tx