# -*- coding: utf-8 -*-
"""
@author: Mohamad Senno
Inputs: Data: nonempty np.array(), Nzero: real number positive
Outputs: Array of complex numbers
Behavior: Adds a Complex Circular Gaussian noise with variance Nzero to the Data input. Returns the output with the added noise.
If Nzero is negative or the data empty it raises corresponding value errors.
"""

import numpy as np
def AWGN_Channel(Data,Nzero):
    if Nzero < 0:
       raise ValueError("Nzero must be non-negative")   
    N=Data.size
    if N== 0:
       raise ValueError("Empty Data!")
    noise=(np.sqrt(Nzero/2)*(np.random.randn(N) + 1j * np.random.randn(N)))
    return Data+noise


"""
Sample usage
Data=np.zeros(10000,dtype=complex)
Nzero=0.5
response=AWGN_Channel(Data, Nzero)
print(response)

"""
