from scipy.signal import butter, lfilter, freqz,filtfilt,savgol_filter,cheby1,sosfilt,sosfreqz
from scipy.fftpack import fft,fftshift
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
#Fs, audio_data = wavfile.read("/home/adarsh/FM/piano.wav")

"""
plt.plot(audio_data)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency Modulated Signal')
plt.show()

i = np.fft.fft(audio_data)
f_i = np.fft.fftfreq(len(audio_data), d=1/Fs)
plt.plot(f_i, np.abs(i))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Input Signal')
plt.show()
"""

# Sample rate and desired cutoff frequencies (in Hz).
sps = 44100
fc = 1e6
fs = 10*fc
t = np.arange(0,1,1/fs)
msg = np.sin(2*np.pi*200*t)


#df = 1/(N*dt);     # df = 1k Hz
#f = np.arange(N) *df    # freq span 1 MHz    
  
#audio_data = 127*(audio_data.astype(np.int16)/ np.power(2,15))
#ym = audio_data#[:,1] # Audio data

#kf = 1e3               # freq sensitivity
#cumsum = np.cumsum(ym)  # Discrete summation
#b = 3   

plt.plot(t,msg)
plt.show()
y_fm = np.cos((2*np.pi*fc*t) + 1*np.sin(2*np.pi*200*t))
# generate frequency axis
n = np.size(t)
fr = (fs/2)*np.linspace(0,1,int(n/2))
# Compute FFT 
X = fft(y_fm)
X_m = (2/n)*abs(X[0:np.size(fr)])
plt.plot(fr,X_m)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.tight_layout()
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



c= np.cos(2*np.pi*fc*t)


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
cutoff = 200  # desired cutoff frequency of the filter, Hz
fs = 3*(cutoff)

def butter_lowpass(cutoff, fs, order=5):
    #nyq = 0.5 * fs
    #cut = cutoff / nyq
    return butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')

def butter_lowpass_filter(data, cutoff, fs, gain=1, order=5):
    sos = butter_lowpass(cutoff, fs, order=order)
    y = sosfilt(sos, data)
    return y    
	
"""def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
    def cheby_lowpass(cutoff, fs, order=5):
    rp=1
    #nyq = 0.5 * fs
    #cut = cutoff / nyq
    return cheby1(order,rp, cutoff, fs=fs, btype='low', analog=False, output='sos')
def cheby_lowpass_filter(data, cutoff, fs, gain=1, order=5):
    sos = cheby_lowpass(cutoff, fs, order=order)
    y = sosfilt(sos, data)
    return y    
"""
y4 =  butter_lowpass_filter(y3, cutoff, fs, gain=5, order=order)
#y4 =  cheby_lowpass_filter(y3, cutoff, fs, gain=5, order=5)    
z4 = np.fft.fft(y4)
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
#plt.show()

plt.plot(y6)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('output Signal')
plt.show()
 
waveform_quiet = y6 * 0.3

# Adjust amplitude of waveform and write the .wav file.
waveform_integers = np.int16(waveform_quiet * 32767)
write('fm-out.wav', sps, waveform_integers)
   
#wavfile.write('o_out.wav', 1,fs)
   
    
 
