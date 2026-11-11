import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Transmitter.Transmitter import Constellation, Waveform, Transmitter
from Channel.AWGNChannel import AWGN_Channel
from Receiver.receiver import Receiver

st.set_page_config(page_title="Communication Systems Simulator",page_icon="📡",layout="wide")
st.title("📡 Communication Systems Simulator")
st.write("Welcome to the EECE 440 CSS Simulator!")

@st.dialog("Simulation Parameters")
def simulation_popup():
    st.write("Let's get you started! What channel do you wish to test today?")
    channel = st.selectbox("Channel Model",["AWGN", "Rayleigh", "Rician","OFDM","Multi-Tap Channel"])
    numtx=st.number_input("Number of Transmitting Antennas",min_value=1,value=1,step=1)
    numrx=st.number_input("Number of Receiving Antennas",min_value=1,value=1,step=1)
    if st.button("Confirm Parameters"):
        for key in ["channel","numtx","numrx","awgn_configured","transmitters","receivers","bits","samples","received","decoded_bits","ber","H"]:
            st.session_state.pop(key,None)
        st.session_state.channel=channel
        st.session_state.numtx=numtx
        st.session_state.numrx=numrx
        st.rerun()

if "initialized" not in st.session_state:
    st.session_state.initialized=True

if st.button("New Simulation"):
    simulation_popup()

if "channel" in st.session_state:
    st.subheader("Current Simulation Configuration")
    st.write(f"Channel: {st.session_state.channel}")
    st.write(f"Transmitting Antennas: {st.session_state.numtx}")
    st.write(f"Receiving Antennas: {st.session_state.numrx}")

    if st.button("Close Simulation"):
        for key in ["channel","numtx","numrx","awgn_configured","transmitters","receivers","bits","samples","received","decoded_bits","ber","H","constellation","M","A","N","N0","c"]:
            st.session_state.pop(key,None)
        st.rerun()

@st.dialog("AWGN Simulation Parameters")
def awgn_popup():
    st.write("Please select AWGN Channel parameters")
    constellation=st.selectbox("Constellation Type",["PAM","QAM","PSK"])
    if constellation=="PAM":
        M_options=[2,4,8,16,32,64,128,256]
    elif constellation=="QAM":
        M_options=[4,16,64,256,1024]
    else:
        M_options=[2,4,8,16,32,64,128,256]
    M=st.selectbox("Modulation Order (M)",M_options)
    A=st.number_input("Average Symbol Energy (A)",min_value=0.000001,value=1.0,step=0.5)
    st.divider()
    N=st.number_input("Message Length (number of bits per transmitter)",min_value=1,value=64,step=1)
    N0=st.number_input("Noise Power Spectral Density (N0)",min_value=0.0,value=0.1,step=0.01,format="%.6f")
    if st.button("Start AWGN Simulation"):
        k=int(np.log2(M))
        if N%k!=0:
            st.error(f"Message length must be divisible by {k} for M={M}. Please adjust N.")
        else:
            st.session_state.constellation=constellation
            st.session_state.M=M
            st.session_state.A=A
            st.session_state.N=N
            st.session_state.N0=N0
            st.session_state.awgn_configured=True
            for key in ["transmitters","receivers","bits","samples","received","decoded_bits","ber","H"]:
                st.session_state.pop(key,None)
            st.rerun()

if st.session_state.get("channel")=="AWGN" and not st.session_state.get("awgn_configured",False):
    awgn_popup()

if st.session_state.get("channel")=="AWGN" and st.session_state.get("awgn_configured",False):

    if "transmitters" not in st.session_state:

        c=Constellation(
            st.session_state.constellation,
            M=st.session_state.M,
            A=st.session_state.A
        )

        transmitters=[
            Transmitter(c)
            for _ in range(st.session_state.numtx)
        ]

        receivers=[
            Receiver(c)
            for _ in range(st.session_state.numtx)
        ]

        st.session_state.transmitters=transmitters
        st.session_state.receivers=receivers
        st.session_state.c=c

        st.success(f"{len(transmitters)} transmitters successfully initialized!")
        st.success(f"{len(receivers)} receivers successfully initialized!")

    if st.button("Run AWGN Simulation"):

        transmitters=st.session_state.transmitters
        receivers=st.session_state.receivers
        N=st.session_state.N
        N0=st.session_state.N0
        numtx=st.session_state.numtx
        numrx=st.session_state.numrx

        bits=[
            np.random.randint(0,2,N)
            for _ in range(numtx)
        ]

        samples=np.array([
            transmitters[i].run(bits[i],N)
            for i in range(numtx)
        ])

        H=np.exp(
            1j*2*np.pi*np.outer(np.arange(numrx),np.arange(numtx))/numrx
        )/np.sqrt(numrx)

        received=AWGN_Channel(H@samples,N0)

        detected=np.linalg.pinv(H)@received

        decoded_bits=[
            receivers[i].decode(detected[i])
            for i in range(numtx)
        ]

        ber=[
            receivers[i].ber(bits[i],decoded_bits[i])
            for i in range(numtx)
        ]

        st.session_state.bits=bits
        st.session_state.samples=samples
        st.session_state.received=received
        st.session_state.decoded_bits=decoded_bits
        st.session_state.ber=ber
        st.session_state.H=H

        st.success("AWGN simulation completed!")

    if "received" in st.session_state:

        st.subheader("Simulation Results")

        if st.session_state.numrx<st.session_state.numtx:
            st.warning("There are fewer receiving antennas than transmitting antennas. The independent streams cannot generally be separated reliably.")

        st.subheader("Transmitted and Received Bits")

        table="<div style='overflow-x:auto;width:100%;'><table style='border-collapse:collapse;white-space:nowrap;width:max-content;'>"

        table+="<tr><th style='padding:8px;border:1px solid #888;'>Transmitter</th><th style='padding:8px;border:1px solid #888;'>Signal</th>"

        for j in range(st.session_state.N):
            table+=f"<th style='padding:8px;border:1px solid #888;'>{j+1}</th>"

        table+="</tr>"

        for i in range(st.session_state.numtx):

            table+=f"<tr><td rowspan='2' style='padding:8px;border:1px solid #888;'>Transmitter {i+1}</td>"
            table+="<td style='padding:8px;border:1px solid #888;'>Transmitted</td>"

            for j in range(st.session_state.N):
                table+=f"<td style='padding:8px;border:1px solid #888;text-align:center;'>{int(st.session_state.bits[i][j])}</td>"

            table+="</tr><tr>"
            table+="<td style='padding:8px;border:1px solid #888;'>Received</td>"

            for j in range(st.session_state.N):
                original=int(st.session_state.bits[i][j])
                decoded=int(st.session_state.decoded_bits[i][j])

                if original==decoded:
                    color="#15803d"
                else:
                    color="#dc2626"

                table+=f"<td style='padding:8px;border:1px solid #888;text-align:center;background-color:{color};color:white;font-weight:bold;'>{decoded}</td>"

            table+="</tr>"

        table+="</table></div>"

        st.markdown(table,unsafe_allow_html=True)

        st.subheader("Bit Error Rates")

        for i in range(st.session_state.numtx):
            st.metric(f"BER - Transmitter {i+1}",f"{st.session_state.ber[i]:.6f}")

        st.metric("Average BER",f"{np.mean(st.session_state.ber):.6f}")