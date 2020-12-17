import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


import numpy as np 
from scipy.io import wavfile
from scipy import signal

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def match_filter(x,template):
    det = np.convolve(x,template[::-1],mode="same")
    det = np.square(normalize_signal(det))
    #Align to beginning of template in x
    det = shift_array(det,-len(template)/2)
    return det

def pyramid(x,nlayers=4,ksize=5):
    layers = []

    for i in range(nlayers-1):
        x = gaussian_blur(x,ksize=ksize)
        x = x[::2]
        layers.append(x)

    return layers

def gaussian_blur(x,ksize=5,sigma=1,dim=1):
    #Generate gaussian kernel
    g = np.zeros(ksize)
    for i in range(ksize):
        g[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((int(ksize/2) - i) ** 2) / (2 * sigma**2))

    #Normalize gaussian kernel [0,1)
    g = g / np.sum(g)

    #Blur by convolution
    if dim==1: #One dimensional
        x = np.convolve(x,g,mode="same") #same=len(x) is same post convolution
    elif dim==2: #Two dimensional
        x = cv2.filter2D(x, -1, g)          #Convolution since gaussian is same after flipping LR/UD
        x = cv2.filter2D(x, -1, g.T)

    return x

def shift_array(arr, num, fill_value=0):
    num = int(num)
    #Source: https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def normalize_signal(x):
    #Remove bias
    x = x - x.mean()
    
    #Normalize
    x = x / x.max()

    return x

if __name__ == "__main__":
    soundfiles_path = "soundfiles/"
    soundfile_1 = "ryan.wav"
    soundfile_2 = "sentence_with_ryan.wav"

    #Load in template and original
    fs,template = wavfile.read(soundfiles_path + soundfile_1)
    _,x = wavfile.read(soundfiles_path + soundfile_2)

    template = butter_lowpass_filter(template,2500,fs,order=6)
    #template = signal.get_window("hanning",len(template)) * template

    t_template = np.linspace(0, len(template)/fs,num=len(template))
    t_x = np.linspace(0, len(x)/fs,num=len(x))

    #Create pyramids
    pyramid_template = pyramid(template,ksize=11)
    pyramid_x = pyramid(x,ksize=11)

    #Apply match filter per resolutions
    for tt,xx in zip(pyramid_template,pyramid_x):
        r = match_filter(xx,tt)

        f,(ax1,ax2,ax3) = plt.subplots(3)
        ax3.plot(np.linspace(0,len(r)/fs,num=len(r)),r)
        ax2.plot(np.linspace(0,len(tt)/fs,num=len(tt)),tt)
        ax1.plot(np.linspace(0, len(xx)/fs,num=len(xx)),xx)
        plt.show()

        


def useless():
 #Read in sound files
    
    soundfile_2 = "nest.wav"
    soundfile_3 = "sentence_ryan_only.wav"
    fs, template = wavfile.read(soundfiles_path + soundfile_1)
    _, x2 = wavfile.read(soundfiles_path + soundfile_2)
    _, x3 = wavfile.read(soundfiles_path + soundfile_3)

    t = np.linspace(0, len(template) / fs, num=len(template))
    t2 = np.linspace(0, len(x2) / fs, num=len(x2))
    t3 = np.linspace(0, len(x3) / fs, num=len(x3))

    x3 = np.zeros(len(x3))
    x3[10000:10000+len(template)] = template

    fg, (ax1,ax2,ax3,ax4) = plt.subplots(4)
    plt.subplots_adjust(hspace=0.5)
    ax1.plot(t,template)
    ax1.set_title("IGNORE ME Template ('ryan')",loc="left")
    ax2.plot(t2,x2)
    ax2.set_title("Utterance: {}".format(soundfile_2), loc="left")


    #Match filter
    r = matched_filter(x3,template)
    
    #plot
    #ax4.plot(t3,r)
    #ax4.set_title("Matched filter output",loc="left")


    #Create vowel template
    #50-200Hz sine wave --> probably more like voicing bar detection
    steps = 10
    startf = 50
    stopf = 200
    freqs = np.linspace(startf,stopf,num=steps)


    #Create sine wave with frequencies given by freqs
    increase = 1.1
    sine_length = int((startf / 1000) * fs * increase)
    voicing_t = np.arange(sine_length)
    voicing_template = np.zeros(sine_length)
    for f in freqs:
        voicing_template += np.sin(2 * np.pi * f * voicing_t / fs)

    #plt.plot(voicing_t,voicing_template)
    #plt.show()

    #Voicing match filter
    r_voicing = matched_filter(x2,voicing_template)

    ax3.plot(t2,r_voicing)
    ax3.set_title("Match Filter Output")
    #Align
    r_voicing  = np.roll(r_voicing,-len(voicing_template))

    ax4.plot(t2,r_voicing)
    ax4.set_title("Match Filter Output Aligned")
    plt.show()