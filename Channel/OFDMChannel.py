# -*- coding: utf-8 -*-
"""
@author: Mohamad Senno
This is the OFDM Code. It also uses the LTap channel implementation from before.
Note the DFT and IDFT conventions, be careful of the sqrt(N) factor being used.
The channel takes input Nzero for variance of the additive noise, L for the taps and symbols
IMPORTANT: Check the sample equalization code provided, if you define the DFT of h as I do no need to rescale by sqrt(N)

"""

import numpy as np

def IDFT(symb):
    N=len(symb)
    n=np.arange(N)
    k=np.arange(N)
    return np.array([np.sum(symb*np.exp(2j*np.pi*k*i/N))/np.sqrt(N) for i in n])

def DFT(symb):
    N=len(symb)
    n=np.arange(N)
    k=np.arange(N)
    return np.array([np.sum(symb*np.exp(-2j*np.pi*n*i/N))/np.sqrt(N) for i in k])

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

def OFDM(symb,L,Nzero):
    inv=IDFT(symb)
    withcyclpref=np.concatenate((inv[-L:],inv))
    outwithpref,h=LTapChannel(L,withcyclpref,Nzero)
    outnopref=outwithpref[L:L+symb.size]
    return DFT(outnopref),h



"""
np.random.seed(42)
N = 64
L = 4
Nzero=0
symb=(2*np.random.randint(0, 2, N)-1+ 1j*(2*np.random.randint(0, 2, N)-1))/np.sqrt(2)
received, h = OFDM(symb, L, Nzero)
H=np.array([np.sum(h*np.exp(-2j*np.pi*np.arange(L)*k/N))for k in range(N)])
detected=received/H
print("Max error:", np.max(np.abs(detected-symb)))
print("Transmitted:", symb[:5])
print("Detected:", detected[:5])
print("Channel taps:",h)
"""
