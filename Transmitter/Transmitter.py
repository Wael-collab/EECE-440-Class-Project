import numpy as np
import math

class Constellation:
    def __init__(self, name, M=None, d=1.0, mapping=None, A=1):

        self.name = name

        if name.upper() == "PAM":
            mapping = self._build_pam(M, d)
        elif name.upper() == "QAM":
            mapping = self._build_qam(M, d)
        elif name.upper() == "PSK":
            mapping = self._build_psk(M)
        else:
            if mapping is None:
                raise ValueError(
                    f"'{name}' is not a built-in constellation (PAM, QAM, PSK) "
                    "and no dictionary was provided."
                )
        self._set_mapping(mapping,A)

    def _set_mapping(self, mapping,A):
        if A <= 0:
            raise ValueError("A (average energy) must be positive.")
        if not isinstance(mapping, dict) or len(mapping) < 2:
            raise ValueError("Mapping must be a dict with at least 2 entries.")

        if any(not isinstance(l, str) or set(l) - {"0", "1"} for l in mapping):
            raise ValueError("Labels must be strings of 0s and 1s.")

        lengths = {len(l) for l in mapping}
        if len(lengths) != 1:
            raise ValueError("All labels must have the same length.")
        k = lengths.pop()

        if len(mapping) != 2 ** k:
            raise ValueError(f"Expected {2**k} labels of length {k}, got {len(mapping)}.")

        table = np.zeros(2 ** k, dtype=complex)
        for label, symbol in mapping.items():
            table[int(label, 2)] = symbol

        if len(np.unique(np.round(table, 12))) != len(table):
            raise ValueError("Two labels are mapped to the same point.")
        
        avg_energy = np.mean(np.abs(table) ** 2)
        if avg_energy == 0:
            raise ValueError("Constellation has zero energy.")

        table = table / np.sqrt(avg_energy)      # step 1: unit average energy
        table = np.sqrt(A) * table               # step 2: average energy A

        self.A = A
        self.bits_per_symbol = k
        self.M = 2 ** k
        self.table = table
        self.mapping = {l: self.table[int(l, 2)] for l in mapping}
        
    def _build_pam(self, M, d):
        #This checks if M<2 and if M is not a power of two, in this case report an error.
        if M is None or M < 2 or (M & (M - 1)) != 0:
            raise ValueError("M-PAM requires M to be a power of 2.")
        if d <= 0:
            raise ValueError("d must be positive.")
        k = int(np.log2(M))
        mapping = {}
        for m in range(M):
            #This generates the gray code, so that we can have the set of all binary strings of length k
            gray = m ^ (m >> 1) 
            #This pads zeros so that all the binary strings will have the same length                     
            label = format(gray, f"0{k}b")           
            mapping[label] = d * (2 * m - (M - 1)) / 2
        return mapping

    def _build_qam(self, M, d):
        #This checks if M<2 and if M is not a power of two, in this case report an error.
        if M is None or M < 4 or (M & (M - 1)) != 0:
            raise ValueError("M-QAM requires M to be a power of 2.")

        L = math.isqrt(M)                  
        if L * L != M:
            raise ValueError("M-QAM requires M to be a perfect square.")

        pam = self._build_pam(L, d)     

        mapping = {}
        for label_i, x_i in pam.items():
            for label_q, x_q in pam.items():
                mapping[label_i + label_q] = x_i + 1j * x_q
        return mapping

    def _build_psk(self, M):
        #This checks if M<2 and if M is not a power of two, in this case report an error.
        if M is None or M < 2 or (M & (M - 1)) != 0:
            raise ValueError("M-PSK requires M to be a power of 2.")

        k = int(np.log2(M))
        #In each step, we rotate the point on the PSK to get all the points on the circle.
        step = np.exp(1j * 2 * np.pi / M)    
        #We start at e^{j0}
        point = 1 + 0j                       
        mapping = {}
        for m in range(M):
            gray = m ^ (m >> 1)
            label = format(gray, f"0{k}b")
            mapping[label] = point
            point = point * step             
        return mapping


class Waveform:
    """Convert constellation symbols to/from a single-carrier or OFDM waveform.

    A single-carrier transmission is one block; OFDM uses blocks of
    n_subcarriers symbols. A cyclic prefix copies the end of each block.
    """

    def __init__(self, name="SC", n_subcarriers=None, cyclic_prefix=0):
        aliases = {
            "SC": "SC",
            "SINGLE_CARRIER": "SC",
            "SINGLE-CARRIER": "SC",
            "OFDM": "OFDM",
        }
        key = str(name).upper()
        if key not in aliases:
            raise ValueError("Waveform must be 'SC' or 'OFDM'.")

        self.name = aliases[key]
        self.n_subcarriers = None
        if (isinstance(cyclic_prefix, (bool, np.bool_))
                or not isinstance(cyclic_prefix, (int, np.integer))
                or cyclic_prefix < 0):
            raise ValueError("cyclic_prefix must be a non-negative integer.")
        self.cyclic_prefix = int(cyclic_prefix)
        if self.name == "OFDM":
            if (not isinstance(n_subcarriers, (int, np.integer))
                    or n_subcarriers <= 0):
                raise ValueError("n_subcarriers must be a positive integer for OFDM.")
            if self.cyclic_prefix >= n_subcarriers:
                raise ValueError("cyclic_prefix must be less than n_subcarriers.")
            self.n_subcarriers = int(n_subcarriers)

    def modulate(self, symbols):
        """Turn a 1-D symbol stream into transmit samples."""
        symbols = np.asarray(symbols, dtype=complex).ravel()
        if symbols.size == 0:
            raise ValueError("symbols must be non-empty.")
        if self.name == "SC":
            if self.cyclic_prefix > symbols.size:
                raise ValueError("cyclic_prefix cannot exceed the number of symbols.")
            time_blocks = symbols.reshape(1, -1)
        else:
            n = self.n_subcarriers
            if symbols.size % n != 0:
                raise ValueError(
                    "The number of symbols must be divisible by n_subcarriers for OFDM."
                )
            blocks = symbols.reshape(-1, n)
            time_blocks = np.fft.ifft(blocks, axis=1) * np.sqrt(n)
        if self.cyclic_prefix:
            prefix = time_blocks[:, -self.cyclic_prefix:]
            time_blocks = np.concatenate((prefix, time_blocks), axis=1)
        return time_blocks.reshape(-1)

    def demodulate(self, samples):
        """Recover constellation symbols from received waveform samples."""
        samples = np.asarray(samples, dtype=complex).ravel()
        if samples.size == 0:
            raise ValueError("samples must be non-empty.")
        if self.name == "SC":
            if samples.size <= self.cyclic_prefix:
                raise ValueError("Received samples must include data after the cyclic prefix.")
            return samples[self.cyclic_prefix:].copy()

        block_length = self.n_subcarriers + self.cyclic_prefix
        if samples.size % block_length != 0:
            raise ValueError("The received samples must contain whole OFDM blocks.")
        blocks = samples.reshape(-1, block_length)
        if self.cyclic_prefix:
            blocks = blocks[:, self.cyclic_prefix:]
        return (np.fft.fft(blocks, axis=1) / np.sqrt(self.n_subcarriers)).reshape(-1)


class Transmitter:
    def __init__(self, constellation, waveform=None):
        """
        constellation : a Constellation object (already normalized and
                        scaled to its energy A).
        waveform : a Waveform object. Defaults to single-carrier transmission.
        """
        if not isinstance(constellation, Constellation):
            raise TypeError("constellation must be a Constellation object.")
        if waveform is None:
            waveform = Waveform("SC")
        if not isinstance(waveform, Waveform):
            raise TypeError("waveform must be a Waveform object.")

        self.constellation = constellation
        self.waveform = waveform
        self.k = constellation.bits_per_symbol
        self.M = constellation.M
    
    def run(self, bits, N):
        if not isinstance(N, (int, np.integer)) or N <= 0:
            raise ValueError("N must be a positive integer.")
        if N % self.k != 0:
            raise ValueError(
                f"N = {N} is not divisible by log2(M) = {self.k}, "
                "so the bits cannot be mapped to whole symbols."
            )

        bits = np.asarray(bits)
        if bits.ndim != 1:
            raise ValueError("bits must be a 1-D array.")
        if not np.issubdtype(bits.dtype, np.integer):
            raise TypeError("bits must be an array of integers (0 and 1).")
        if len(bits) != N:
            raise ValueError(f"Expected {N} bits, got {len(bits)}.")

        return self.modulate(bits)

    def modulate(self, bits):
        bits = np.asarray(bits, dtype=int).ravel()

        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("Input must contain only 0s and 1s.")
        if len(bits) % self.k != 0:
            raise ValueError(f"Number of bits ({len(bits)}) must be a "
                            f"multiple of {self.k}.")

        symbols = []
        for i in range(0, len(bits), self.k):
            chunk = bits[i:i + self.k]                    
            label = "".join(str(b) for b in chunk)        
            symbols.append(self.constellation.mapping[label])

        return self.waveform.modulate(np.array(symbols))

if __name__ == "__main__":
    N = 64
    bits = np.random.randint(0, 2, N)
    print("bits:", bits)

    for name in ["PAM", "QAM", "PSK"]:
        c = Constellation(name, M=16, A=4)
        tx = Transmitter(c)
        samples = tx.run(bits, N)
        print(bits)
        print(samples)
