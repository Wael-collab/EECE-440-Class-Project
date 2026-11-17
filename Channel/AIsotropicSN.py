"""
@author: Dongmin Park, Karim Ezzeddine
Isotropic complex symmetric alpha-stable (S-alpha-S) noise, "sub-Gaussian" model.

In the Channel/AISN.py model, the real and imaginary parts of the noise
are two INDEPENDENT symmetric alpha-stable variables.
For alpha < 2 in AISN, the noise is not rotation invariant.
The standard complex S-alpha-S model is isotropic
and is built from a Gaussian scaled by a positive stable variable:

N = sqrt(A) * G,    G ~ CN(0, 2 gamma^2 * I) (re, im each N(0, 2 gamma^2))
                    where I is the 2 by 2 identity matrix
                    A ~ positive (alpha/2)-stable as given in Nolan.
                    and A and G are independent.
"""
import numpy as np
import aub_htp as ht

def Isotropic_Alpha_Stable_Channel(Data, alpha, gamma):
    """isotropic symmetric alpha-stable noise"""

    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    Data = np.asarray(Data)

    N = Data.shape
    if Data.size == 0:
        raise ValueError("Empty Data!")

    # A based on Nolan's form with beta = 1, loc = 0 and scale = cos(pi*alpha/4)^(2/alpha)
    if alpha == 2:
        A = np.ones(N)  # alpha/2 = 1 is degenerate: A = 1, the noise is Gaussian
    else:
        a = alpha / 2
        A = ht.alpha_stable.rvs(alpha=a,beta=1,loc=0,scale=np.cos(np.pi*a/2)**(1/a),size=N)

    # Circular Gaussian part: real and imaginary parts each N(0, 2 gamma^2)
    G = np.sqrt(2) * gamma * (np.random.randn(*N) + 1j * np.random.randn(*N))

    noise = np.sqrt(A) * G

    return Data + noise

"""
# Sample usage
# SISO
Data = np.zeros(10000, dtype=complex)

# MIMO: Nt transmit antennas, Nr receive antennas, N symbols
# Data = np.zeros((Nr, 10000), dtype=complex)

response = Isotropic_Alpha_Stable_Channel(Data, alpha=1.5, gamma=1)
print(response)
"""