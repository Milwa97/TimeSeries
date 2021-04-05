#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq

import scipy as sp
from scipy.signal import periodogram

import warnings
warnings.filterwarnings('ignore')

π = np.pi

############################################################################################################
############################################################################################################
## Helper functions: getting data

def get_signal(path, number_of_points):

    data = np.genfromtxt(path, delimiter='\t').T
    time = data[0] *512
    signal = data[1]
    time = time[:number_of_points]
    signal = signal[:number_of_points]
    N = len(time)                           ## number of points
    Δt = time[1]-time[0]                    ## timestep = how often signal is sampled
    f_Nyq = 1/(2*Δt)                        ## Nyquist frequency
    print("number of points {:} Nyquist frequency: {:}".format(N, f_Nyq))   
    return time, signal, N, Δt, f_Nyq



def get_power_spectrum(signal, N,  Δt, norm =1.0):
    
    signal_fft = fft(signal, norm = 'ortho')        ###  DFT  
    frequency = fftfreq(N, Δt)[:N//2]               ###  only non-negative frequenties

    power_spectrum = np.zeros(N//2)
    power_spectrum[0] = np.abs(signal_fft[0])**2
    power_spectrum[-1] = np.abs(signal_fft[N//2])**2
    for n in range(1, N//2):
        power_spectrum[n]= np.abs(signal_fft[n])**2 + np.abs(signal_fft[-n])**2
    
    power_spectrum = power_spectrum/norm
    
    return frequency, power_spectrum



############################################################################################################
############################################################################################################
## Helper functions: plotting data


def plot_signal(time, signal):
    
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    axs[0].scatter(time, signal, c='darkblue', s=1)
    axs[1].plot(time, signal, c='purple')

    for ax in axs:
        ax.set_xlabel('time')
        ax.set_ylabel('signal')
        ax.grid(True)
    plt.show()




def plot_power_spectrum(frequency, power_spectrum):
    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    axs[0].stem(frequency, power_spectrum, linefmt='grey', markerfmt='.')
    axs[1].stem(frequency, power_spectrum, linefmt='grey', markerfmt='.')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    axs[1].set_ylim(1e-6, 1e4)

    for ax in axs:
        ax.set_xlabel('frequency')
        ax.set_ylabel('Power spectrum density')
        ax.grid(True)
        ax.set_ylim(1e-6, 1e4)

    plt.show()
    
    
    
def compare_signal(time, signal, signal_filtred, title = "signal filtered using Wiener filter"):
    
    fig, axs = plt.subplots(2, 1, figsize=(15,8), sharex = True)
    axs[0].plot(time, signal, c='darkblue')
    axs[1].plot(time, signal_filtred, c='purple')

    axs[0].set_title("original signal")
    axs[1].set_title(title)
    axs[1].set_xlabel('time')

    for ax in axs:
        ax.set_ylabel('signal')
        ax.grid(True)
    plt.tight_layout();


def compare_pwd(frequency, power_spectrum, power_spectrum_filtred, y_min =1e-6, y_max = 1e4 ):
       
    fig, axs = plt.subplots(1, 2, figsize=(15,5), sharex = True)

    axs[0].stem(frequency, power_spectrum, linefmt='lightgrey', markerfmt='.')
    axs[1].stem(frequency, power_spectrum_filtred, linefmt='lightgrey', markerfmt='.')

    axs[0].set_title("Power Spectrum of original signal")
    axs[1].set_title("Power Spectrum of filtered signal")
    axs[1].set_xlabel('frequency')

    for ax in axs:
        ax.set_ylabel('Power spectrum density') 
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
    plt.tight_layout();








