import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


import numpy as np 
from scipy.io import wavfile
from scipy import signal


if __name__ == "__main__":
    #Load in sound file
    soundfiles_path = "soundfiles/"
    soundfile = "ei.wav"

    fs, x = wavfile.read(soundfiles_path + soundfile)
    
    time_series = np.linspace(0, len(x) / fs, num=len(x))

    #Simple cleaning/enhancement
    y_offset = np.mean(x)
    x = x - y_offset

    #plot time-series and spectrogram
    vowel_period_length = 0.075
    NFFT = int(0.01 * fs)
    noverlap = int(0.005 * fs)
    nperseg = int((NFFT + noverlap) / 2)
    print(NFFT, noverlap, nperseg)
    f, t, Sxx = signal.spectrogram(x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=NFFT,
        window=signal.get_window('hanning', nperseg),
        detrend=False)
    
    #Upsample and guassian smooth?
    

    fig, (ax1,ax2) = plt.subplots(2)
    ax1.set(xlabel='Time [sec]')
    ax1.plot(time_series, x)

    ax2.pcolormesh(t, f, Sxx, cmap=cm.gray_r, shading='gouraud', norm=LogNorm())
    ax2.set(xlabel='Time [sec]', ylabel='Frequency [Hz]')
    ax2.set_ylim(0,5000)
    plt.show()