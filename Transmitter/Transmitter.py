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

class Transmitter:
    def __init__(self, constellation):
        """
        constellation : a Constellation object (already normalized and
                        scaled to its energy A).
        """
        if not isinstance(constellation, Constellation):
            raise TypeError("constellation must be a Constellation object.")

        self.constellation = constellation
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

        return np.array(symbols)

N = 64   
bits = np.random.randint(0, 2, N)
print("bits:", bits)

for name in ["PAM", "QAM", "PSK"]:
    c = Constellation(name, M=16,A=4)
    tx = Transmitter(c)
    symbols = tx.run(bits, N)
    print(bits)
    print(symbols)