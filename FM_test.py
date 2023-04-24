from scipy.signal import butter, lfilter, freqz,filtfilt,savgol_filter,cheby1,sosfilt,sosfreqz
from scipy.fftpack import fft,fftshift
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

Fs, audio_data = wavfile.read("/home/imran/nik/Sound.wav")

"""
plt.plot(audio_data)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency Modulated Signal')
plt.show()
"""
i = np.fft.fft(audio_data)
f_i = np.fft.fftfreq(len(audio_data), d=1/Fs)
plt.plot(f_i, np.abs(i))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Input Signal')
#plt.savefig('/home/imran/nikhil bhai/input message.pdf')
plt.show()


# Sample rate and desired cutoff frequencies (in Hz).

fc = 100e6
fs = 250e6

N = len(audio_data)
dt = 1e-3/N             # time span 1 msec
t = np.arange(N)    
t = t*dt

#df = 1/(N*dt);     # df = 1k Hz
#f = np.arange(N) *df    # freq span 1 MHz    
  
audio_data = 127*(audio_data.astype(np.int16)/ np.power(2,15))
ym = audio_data#[:,1] # Audio data

kf = 20              # freq sensitivity
cumsum = np.cumsum(ym)  # Discrete summation
    
c = np.cos(2*np.pi*fc*t)
y_fm = np.cos(2*np.pi*fc*t + kf*cumsum*(1/Fs))
    
z = np.fft.fft(y_fm)
f_2 = np.fft.fftfreq(len(y_fm), d=dt)

plt.plot(f_2,abs(z))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.title('FM Signal')
plt.legend(loc='best')
plt.show()


# Bandpass Filter 
def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, gain=1, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y
"""   
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, gain=1, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y    

def cheby_bandpass(lowcut, highcut, fs, order=5):
    rp=6
    b, a = cheby1(order,rp, [lowcut, highcut],fs=fs, btype='band')
    return b, a
def cheby_bandpass_filter(data, lowcut, highcut, fs, gain=1, order=5):
    b, a = cheby_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y"""


lowcut = fc - 100e3
highcut = fc + 100e3
# Plot the frequency response of BPF
plt.figure(1)
plt.clf()
order = 5
for order in [order]:
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    #b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #b, a = cheby_bandpass(lowcut, highcut, fs, order=order)
    w, h = sosfreqz(sos,fs=fs, worN=2000)
    #w, h = freqz(b,a,fs=fs, worN=2000)
    plt.plot(w, 350*abs(h), label="order = %d" % order)


y2 = butter_bandpass_filter(y_fm, lowcut, highcut, fs, gain=5, order=7)
#y2 = cheby_bandpass_filter(y_fm, lowcut, highcut, fs, gain=5, order=order)
z2 = np.fft.fft(y2)
f_bp = np.fft.fftfreq(len(y2), d=1/fs)
plt.plot(f_bp,abs(z2))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.title('BPF Output')
plt.show()






y3 = y2*c
    
z3 = np.fft.fft(y3)
f_3 = np.fft.fftfreq(len(y3), d=1/fs)
plt.plot(f_3,abs(z3))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.title('Mixer Output')
plt.show()








#Lowpass Filter
cutoff = 2e3  # desired cutoff frequency of the filter, Hz
fs = 3*(cutoff)

"""def butter_lowpass(cutoff, fs, order=5):
    #nyq = 0.5 * fs
    #cut = cutoff / nyq
    return butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')

def butter_lowpass_filter(data, cutoff, fs, gain=1, order=5):
    sos = butter_lowpass(cutoff, fs, order=order)
    y = sosfilt(sos, data)
    return y    
	
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y"""
    
def cheby_lowpass(cutoff, fs, order=5):
    rp=1
    #nyq = 0.5 * fs
    #cut = cutoff / nyq
    return cheby1(order,rp, cutoff, fs=fs, btype='low', analog=False, output='sos')
def cheby_lowpass_filter(data, cutoff, fs, gain=1, order=5):
    sos = cheby_lowpass(cutoff, fs, order=order)
    y = sosfilt(sos, data)
    return y    

#y4 =  butter_lowpass_filter(y3, cutoff, fs, gain=5, order=order)
y4 =  cheby_lowpass_filter(y3, cutoff, fs, gain=5, order=5)    
z4 = np.fft.fft((y4)/N)
f_4 = np.fft.fftfreq(len(y4), d=1/fs)
plt.plot(f_4,abs(z4))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.title('LPF Output')
plt.show()





y5 = np.arccos(y4)    # cos inverse of LPF output





# Differentiation of cos inverse
i=0
y6 = np.zeros(len(t))

while i< len(t)-1:
      y6[i] = (y5[i+1] - y5[i]) / (t[i+1] - t[i])
      i = i+1

"""
# Set window length and polynomial order
window_length = 51
polyorder = 2

# Apply Savitzky-Golay filtering to y5 to obtain y6
y6 = savgol_filter(y5, window_length, polyorder, deriv=1, delta=t[1]-t[0], mode='mirror')"""

z5 = np.fft.fft(y6)
f_5 = np.fft.fftfreq(len(y6), d=1/fs)
plt.plot(f_5,abs(z5))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.title('Final Output Signal')
plt.show()

plt.plot(y6)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency Modulated Signal')
#plt.show()
    
wavfile.write('signal_out.wav', Fs, y6)
   
    
 
