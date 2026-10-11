"""
@dongmin park
"""
import numpy as np
import aub_htp as ht

def Alpha_Stable_Channel(Data, alpha, gamma):
    """symmetric alpha-stable noise"""

    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    Data = np.asarray(Data)

    N = Data.size
    if N == 0:
        raise ValueError("Empty Data!")

    # Independent symmetric alpha-stable real and imaginary noise
    noise_real = ht.alpha_stable.rvs(alpha=alpha,beta=0,loc=0,scale=gamma,size=N)
    noise_imag = ht.alpha_stable.rvs(alpha=alpha,beta=0,loc=0,scale=gamma,size=N)
    noise = noise_real + 1j * noise_imag

    return Data + noise

"""
# Sample usage
Data = np.zeros(10000, dtype=complex)
response = Alpha_Stable_Channel(Data, alpha=1.5, gamma=1)
print(response)
"""