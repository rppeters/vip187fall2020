import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm


import numpy as np 
from scipy.io import wavfile
from scipy import signal

def generate_gaussian_kernel(length=3, sigma=1):
    kernel = np.zeros(length)
    for i in range(length):
        kernel[i] = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((int(length/2) - i) ** 2) / (2 * sigma**2))

    kernel = kernel / np.sum(kernel)
    return kernel

def dog_response(x,ksize=3):
    ksize_1 = ksize
    ksize_2 = int(np.rint(ksize_1 * 1.6))
    g1 = generate_gaussian_kernel(length=ksize_1)
    g2 = generate_gaussian_kernel(length=ksize_2)
    

    #Convolve claps with gaussian kernels
    r_g1 = np.convolve(x,g1,mode='same')
    r_g2 = np.convolve(x,g2,mode='same') #same returns r of size x

    #Subtract convolutions of gaussian kernels to get DoG
    DoG = r_g2 - r_g1

    #Find zero crossings of 
    #zero_crossings = np.where(np.diff(np.sign(DoG)))[0]
    return DoG

def moving_average(x, window=5):
    return np.convolve(x, np.ones(window), mode='same') / window

if __name__ == "__main__":
    #Read in sound files
    soundfiles_path = "soundfiles/"
    soundfile = "claps.wav"
    fs, x = wavfile.read(soundfiles_path + soundfile)
    t = np.linspace(0, len(x) / fs, num=len(x))



    #Add noise to time series for testing I guess
    #noise = np.random.normal(0,100,x.shape[0])
    #x = x + noise

    #Absolute value of time series
    x = np.abs(x)
    window_size = int(0.005 * fs)
    x = moving_average(x,window=window_size)

    fig, (ax1,ax2,ax3) = plt.subplots(3)    
    ksizes = np.array([3,5,7])
    for ksize,axis in zip(ksizes,[ax1,ax2,ax3]):
        response = dog_response(x,ksize=ksize)

        print(np.std(response))

        second_d = np.array([1,-2,1])
        #response = np.convolve(response,second_d, mode="same")
        pwidth = 0.0025 * fs
        pprominence = np.std(response)
        pdistance = 0.005 * fs
        peaks, _ = signal.find_peaks(response,prominence=pprominence,width=pwidth,distance=pdistance)


        axis.plot(t,x)
        axis.plot(t,response)
        for peak in peaks:
            axis.plot(peak /fs,x[peak],"x")
        """
        distances = []
        for i in range(int(0.25 * len(peaks)),int(0.75 * len(peaks))):
            distances.append((peaks[i+1] - peaks[i]) / fs )
        
        print("Vocal Fold Vibration: {} Hz".format(1 / np.mean(distances)))
        """
        
            
            

        #axis.plot(peaks,x[peaks],"x")
        axis.set(xlabel="Time (s)", ylabel="Amplitude/Response")
        axis.set_title("Kernel Size={}".format(ksize),loc="left")
        

        """
        #Vocal fold periodic insights for voicing
        smooth = moving_average(response, window=201)
        smooth = np.convolve(smooth,generate_gaussian_kernel(length=5),mode="same")
        axis.plot(t,smooth,"y") """

        

    plt.show()