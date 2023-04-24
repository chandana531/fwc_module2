
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
from math import pi
plt.close('all')

# Generate sine wave
Fs =10000000
t = np.arange(0,1,1/Fs)
f = 1000000

x = np.cos((2*pi*f*t)+1*np.sin(2*pi*20000*t))

plt.subplot(2,1,1)
plt.plot(t,x); plt.title('Sinusoidal Signal');
plt.xlabel('Time(s)'); plt.ylabel('Amplitude')
plt.grid('on')

# generate frequency axis
n = np.size(t)
fr = (Fs/2)*np.linspace(0,1,int(n/2))
# Compute FFT 
X = fft(x)
X_m = (2/n)*abs(X[0:np.size(fr)])

plt.subplot(2,1,2)
plt.plot(fr,X_m); plt.title('Magnitude Spectrum')
plt.xlabel('Frequency(Hz)'); plt.ylabel('Magnitude')
plt.tight_layout()
plt.grid('on')
plt.show()
