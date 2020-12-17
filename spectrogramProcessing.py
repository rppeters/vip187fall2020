import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import argparse

import numpy as np 
from scipy.io import wavfile
from scipy import signal, ndimage

import cv2

def normalize_signal(x):
    #Remove bias
    x = x - x.mean()
    
    #Normalize
    x = x / x.max()

    return x

def calculate_spectrogram(x,fs,window_type="hanning",nfft_coeff=0.01,noverlap_coeff=0.005):
    NFFT = int(0.01 * fs)
    noverlap = int(0.005 * fs)
    nperseg = int((NFFT + noverlap) / 2)
    nperseg=NFFT
    
    print("Spectrogram Parameters:\nNFFT={}\nnOverlap={}\nnperseg={}\n".format(NFFT,noverlap,nperseg))

    #Returns (t,f,Sxx) where:
    #t = time bins
    #f = frequency bins 
    #Sxx = intensity of a frequency at a time (Sxx[t,f])
    return signal.spectrogram(x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=NFFT,
        window=signal.get_window(window_type, nperseg),
        detrend=False)

def gaussian_blur(x,size=5,sigma=1,dim=1):
    #Generate gaussian kernel
    g = np.zeros(size)
    for i in range(size):
        g[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((int(size/2) - i) ** 2) / (2 * sigma**2))

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

def extract_vowel(x,template,fs):
    #Convolve 
    resp = np.convolve(x,template,mode="same")

    #Normalize response to [-1,1) and Square response (raise peaks, lower valleys)
    resp = np.square(normalize_signal(resp))

    #Apply moving average to absolute value of response
    peaks,_ = signal.find_peaks(resp)
  
    #Expand peaks to length of x using interpolation to obtain an envelope of resp to match filter
    resp = np.interp(np.arange(len(resp)),peaks,resp[peaks]) 

    #Smooth
    window = 500
    resp = gaussian_blur(resp,size=9)
    resp = np.convolve(resp, np.ones(window), mode='same') / window

    #Apply threshold
    threshold = 0.15 * resp.max()

    mask = np.ones(len(resp))
    mask[np.where(resp < threshold)] = 0 

    #Align response mask to original by half the length of template due to match filter computation
    mask = shift_array(mask, int(len(template)/2))
    vowel = x * mask

    return vowel

def edge(Sxx, sigma=1):
    Sxx = np.log10(Sxx)
    
    f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)

    #Blur Spectrogram
    gaussian = np.array([2,4,5,4,2,4,9,12,9,4,5,12,15,12,5,4,9,12,9,4,2,4,5,4,2]).reshape(5,5) / 159
    Sxx_blur = cv2.filter2D(Sxx, -1, gaussian) 
    ax1.imshow(cv2.resize(Sxx_blur,(5000,5000),interpolation=cv2.INTER_LINEAR),origin="lower")

    #Find y maxima
    sobel_y_2order = np.array([1,4,6,4,1,0,0,0,0,0,-2,-8,-12,-8,-2,0,0,0,0,0,1,4,6,4,1]).reshape(5,5)
    edges_vert = cv2.filter2D(Sxx_blur, -1, sobel_y_2order)
    #edges_vert = cv2.resize(edges_vert,(5000,5000),interpolation=cv2.INTER_LINEAR)


    #normalize between [0,1]
    edges_vert = edges_vert - edges_vert.min()
    edges_vert = edges_vert / edges_vert.max()
    edges_vert = 1 - edges_vert #invert so formants are higher values (allows for better convolution processing)

    #Threshold (goal=find minimum --> formants
    threshold = 0.6
    edges_vert_copy = np.copy(edges_vert)
    edges_vert[edges_vert < threshold] = 0
    ax2.imshow(edges_vert,origin="lower")


    #Find local maximums column-wise (hysterisis)
    def find_maxima(arr):
        maxes = signal.argrelextrema(arr,np.greater)
        b = np.zeros_like(arr)
        b[maxes] = 1
        return b*arr
    
    edges = np.apply_along_axis(find_maxima, 0, edges_vert)
    ax3.imshow(edges,origin="lower")

    #Testing
    t1 = 0.75   #goal is no "fake"edges
    t2 = 0.6    
    t3 = 0.5
    test1 = np.copy(edges_vert_copy)
    test2 = np.copy(edges_vert_copy)
    test3 = np.copy(edges_vert_copy)
    test1[test1 < t1] = 0
    test2[test2 < t2] = 0
    test3[test3 < t3] = 0
    t1e = np.apply_along_axis(find_maxima, 0, test1)
    t2e = np.apply_along_axis(find_maxima, 0, test2)
    t3e = np.apply_along_axis(find_maxima, 0, test3)
    
    ax4.imshow(t1e)
    ax5.imshow(t2e)
    ax6.imshow(t3e)
    plt.show()


    #Hysterisis
    def connect_edges(strong,weak):
        while(1):
            updated = np.copy(strong)
            flag = False
            rs,cs = np.where(weak!=updated)
            for r2,c2 in zip(rs,cs):
                if strong[r2,c2] == 0 and np.count_nonzero(strong[r2-1:r2+2,c2-1:c2+2]) >= 1:
                    updated[r2,c2] = weak[r2,c2]
                    flag = True

            if not flag: #if no changes on iteration
                break 
            else:
                strong = updated

        return updated
    

    strong = connect_edges(connect_edges(t1e,t2e),t3e)
    plt.imshow(cv2.resize(strong,(5000,5000),cv2.INTER_LINEAR),origin='lower')
    plt.show()

    
    #Use surrounding values to interpolate the fft from low resolution
    rows,cols = np.where(strong != 0) #edge locations
    edges_vert = np.copy(edges_vert_copy)
    freq_bins = edges_vert.shape[0]
    scale_factor = 5000 / freq_bins
    freqs = np.zeros_like(edges_vert)
    
    for r,c in zip(rows,cols):
        formant = edges_vert[r-2:r+3,c]
        formant = formant / formant.sum()


        freq = scale_factor * (freq_bins - r)

        h = np.arange(len(formant)) * scale_factor
        h = h[::-1]  
        h = h - np.median(h)

        if (formant[1] > formant[3]):
            new_freq = freq + np.sum(h[0:2] * formant[0:2])
        elif (formant[1] < formant[3]):
            new_freq = freq + np.sum(h[3:5] * formant[3:5])
        else:
            new_freq = freq + np.sum(h * formant)

        new_freq += scale_factor #first row should be scale_factor not 0

        freqs[r,c] = new_freq 

    for row in freqs:
        for c in row:
            print("{:<4} ".format(int(c)),end='')
        print()

    

    



def create_template(fs,start=50,stop=200,steps=10,cycles=1.1):
    #Create frequencies for the sine wave
    freqs = np.linspace(start,stop,num=steps)

    #Make length to be at least one cycle 
    sine_length = int((start / 1000) * fs * cycles)

    voicing_t = np.arange(sine_length)
    voicing_template = np.zeros(sine_length)
    for f in freqs:
        voicing_template += np.sin(2 * np.pi * f * voicing_t / fs)

    return voicing_template

def display(x,ts,f,t,Sxx,resp=None,save=False):
    #Display spectrogram
    if resp is None:
        ff, (ax1,ax2) = plt.subplots(2)
    else:
        ff, (ax1,ax2,ax3) = plt.subplots(3)
        ax3.set(xlabel="Time [sec]",ylabel="Amplitude")
        ax3.plot(ts,resp)

    ax1.set(xlabel='Time [sec]',ylabel="Amplitude",title="") 
    ax1.plot(ts, x)

    ax2.pcolormesh(t, f, Sxx, cmap=cm.gray_r, shading='gouraud', norm=LogNorm())
    ax2.set(xlabel='Time [sec]', ylabel='Frequency [Hz]')
    ax2.set_ylim(0,5000)

    if save:
        plt.savefig("test.png")
    plt.show()

def check_args(args):
    if args.tstart >= args.tstop:
        print("tstart must be less than tstop")
        exit()

if __name__ == "__main__":
    #Initialize arg parse for command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--file",dest="file",action="store",default="test.wav",
                        help="Image in soundfiles folder to load into program")
    parser.add_argument("--tstart",dest="tstart",action="store",type=int, default=50)
    parser.add_argument("--tstop",dest="tstop",action="store",type=int, default=200)
    parser.add_argument("--tsteps",dest="tsteps",action="store",type=int, default=10)
    parser.add_argument("--tcycles",dest="tcycles",action="store",type=float, default=1.1)
    
    args = parser.parse_args()
    check_args(args)

    #Load Image
    image_dir = "soundfiles/"

    fs,x = wavfile.read(image_dir + args.file)
    x = x[int(0.6*fs):int(1.4*fs)]
    ts = np.linspace(0,len(x)/fs,num=len(x))

    


    #Normalize signal
    x = normalize_signal(x)

    #Process signal
    #Remove Noise 
    x = gaussian_blur(x)

    #Calculate spectrogram
    f, t, Sxx = calculate_spectrogram(x,fs,
        window_type="hanning",
        nfft_coeff=0.01,
        noverlap_coeff=0.005)

    #Get template
    template = create_template(fs,
        start=args.tstart,
        stop=args.tstop,
        steps=args.tsteps,
        cycles=args.tcycles)

    #Match filter
    vowel_in_series = extract_vowel(x,template,fs)

    #Display
    display(x, ts, f, t, Sxx, resp=vowel_in_series)

    #Isolate vowel
    vowel = x[np.where(vowel_in_series != 0)]

    #Calculate vowel spectrogram and display
    f,t,Sxx = calculate_spectrogram(vowel,fs,
        window_type="hanning",
        nfft_coeff=0.1,
        noverlap_coeff=0.005)
    

    v_ts = np.linspace(0,len(vowel)/fs,num=len(vowel))

    #display(vowel,v_ts,f,t,Sxx,save=True)


    #Sxx is from 0-22kHz, crop to 0-5k Hz for speech
    freq_max = f[-1]
    desired_max = 5000
    crop_height = int((desired_max / freq_max) * Sxx.shape[0])
    Sxx = Sxx[0:crop_height,:]

    #Edge Detection
    edge(Sxx)

    


    #wavfile.write("output.wav",fs,np.int16(vowel/np.max(np.abs(vowel)) * 32767))

    


    