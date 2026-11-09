# -*- coding: utf-8 -*-
"""
@author= Mohamad Senno
READ IMPORTANT!
This model assumes a channel with Tapp<Tc and hence h_k do not depend on m
In this model the h_k are sampled according to a CN(0,1) (After normalizing by 1/sqrt(2L)) as in the slides
It takes input symbols, Nzero the variance of the additive noise and L the number of taps
It returns a complex array of transmitted symbols, lenght is not the lenght of the transmitted symbols!
h is also returned for coherent reception.
"""

import numpy as np

def convolve(x,y):
    N=len(x)
    M=len(y)
    z=np.zeros(N+M-1,dtype=complex)
    for i in range(N):
        for j in range(M):
            z[i+j]+=x[i]*y[j]
    return z
def LTapChannel(L,symbols,Nzero):
    h=np.random.randn(L) + 1j * np.random.randn(L)
    h=h/np.sqrt(2*L)
    out=convolve(h,symbols)
    N=out.size
    if Nzero < 0:
       raise ValueError("Nzero must be non-negative")   
    if N== 0:
       raise ValueError("Empty Data!")
    data=out+(np.sqrt(Nzero/2)*(np.random.randn(N) + 1j * np.random.randn(N)))
    return data,h


"""
np.random.seed(42)
symbols=np.random.choice([-1, 1],size=10)
L=7
Nzero=0.1
received, h = LTapChannel(L,symbols,Nzero)
print("Transmitted symbols:",symbols)
print("Channel taps:",h)
print("Received samples:",received)
print("Expected received length:",len(symbols)+L-1)
print("Actual received length:",len(received))
"""