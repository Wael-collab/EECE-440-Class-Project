
# -*- coding: utf-8 -*-
"""
@author= Mohamad Senno
READ IMPORTANT!
This model assumes a channel with Tapp<Tc and hence h_k do not depend on m
In this model the h_k are sampled according to a CN(0,1) (After normalizing by 1/sqrt(2L)) as in the slides
It takes input symbols, Nzero the variance of the additive noise, L the number of taps, Nt the number of transmit antennas and Nr the number of receive antennas
It returns a complex matrix of received symbols, length is not the length of the transmitted symbols!
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

def MIMOLTapChannel(L,symbols,Nzero,Nt,Nr):
    symbols=np.asarray(symbols)
    if Nzero < 0:
       raise ValueError("Nzero must be non-negative")
    if L<=0 or Nt<=0 or Nr<=0:
       raise ValueError("L, Nt and Nr must be positive")
    if symbols.size==0:
       raise ValueError("Empty Data!")
    if symbols.ndim!=2 or symbols.shape[0]!=Nt:
       raise ValueError("Symbols must be a matrix with Nt rows")

    h=np.random.randn(Nr,Nt,L) + 1j * np.random.randn(Nr,Nt,L)
    h=h/np.sqrt(2*L)

    N=symbols.shape[1]+L-1
    out=np.zeros((Nr,N),dtype=complex)

    for r in range(Nr):
        for t in range(Nt):
            out[r,:]+=convolve(h[r,t,:],symbols[t,:])

    data=out+(np.sqrt(Nzero/2)*(np.random.randn(Nr,N) + 1j * np.random.randn(Nr,N)))
    return data,h


"""
np.random.seed(42)
Nt=1
Nr=1
symbols=np.random.choice([-1, 1],size=(Nt,10))
L=7
Nzero=0.1
received, h = MIMOLTapChannel(L,symbols,Nzero,Nt,Nr)
print("Transmitted symbols:",symbols)
print("Channel taps:",h)
print("Received samples:",received)
print("Expected received shape:",(Nr,len(symbols[0])+L-1))
print("Actual received shape:",received.shape)
"""
