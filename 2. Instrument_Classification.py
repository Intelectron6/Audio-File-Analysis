# This program does the following-
# Read each new audio file, finds spectrum and extracts features
# Identifies notes and onsets using signal processing techniques
# Identifies instruments by training an SVM classifier on the dataset created by the previous program 

# Import Modules

import numpy as np
import math
import wave
import os
import struct
import librosa
from scipy.signal import get_window
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Function Name: dft
# Input: time domain signal, window, number of points of DFT
# Output: magnitude spectrum, phase spectrum
# Logic: computing dft using fft function of numpy and extracting magnitude and phase values from it
# Example Call: mX, pX = dtf(x, w, N)
def dft(x, w, N):
    tol = 1e-14
    hN = (N//2)+1                                          
    hM1 = (w.size+1)//2                             
    hM2 = w.size//2                                         
    fftbuffer = np.zeros(N)                              
    w = w / sum(w)                                      
    xw = x*w                                           
    fftbuffer[:hM1] = xw[hM2:]                           
    fftbuffer[-hM2:] = xw[:hM2]        
    X = np.fft.fft(fftbuffer)                               
    absX = abs(X[:hN])                                    
    absX[absX<np.finfo(float).eps] = np.finfo(float).eps   
    mX = 20 * np.log10(absX)                                
    X[:hN].real[np.abs(X[:hN].real) < tol] = 0.0            
    X[:hN].imag[np.abs(X[:hN].imag) < tol] = 0.0                  
    pX = np.unwrap(np.angle(X[:hN]))                     
    return mX, pX

# Function Name: peakDetect
# Input: magnitude spectrum, threshold value
# Output: peak locations
# Logic: selecting range, comparing each value with next and previous values and finding out peaks 
# Example Call: ploc = peakDetect(mX, t)
def peakDetect(mX, t):
    thresh = np.where(np.greater(mX[1:-1],t), mX[1:-1], 0);
    next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     
    prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    
    ploc = thresh * next_minor * prev_minor                 
    ploc = ploc.nonzero()[0] + 1                            
    return ploc

# Function Name: peakInterp
# Input: magnitude spectrum, phase spectrum, peak locations
# Output: interpolated peak locations, interpolated peak magnitude, interpolated peak phase
# Logic: interpolate peak values using parabolic interpolation
# Example Call: iploc, ipmag, ipphase = peakInterp(mX, pX, ploc)
def peakInterp(mX, pX, ploc):
    val = mX[ploc]                                         
    lval = mX[ploc-1]                                      
    rval = mX[ploc+1]                                       
    iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)      
    ipmag = val - 0.25*(lval-rval)*(iploc-ploc)            
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   
    return iploc, ipmag, ipphase

# Function Name: TWM_p
# Input: interpolated frequency, interpolated magnitude, possible f0 candidates
# Output: f0 i.e note
# Logic: implementation of 2-way mismatch algorithm
# Example Call: f0 = TWM_p(ipfreq, ipmag, f0c) 
def TWM_p(pfreq, pmag, f0c):
    p = 0.5                                          
    q = 1.4                                       
    r = 0.5                                        
    rho = 0.33                                      
    Amax = max(pmag)                                
    maxnpeaks = 10                                   
    harmonic = np.matrix(f0c)
    ErrorPM = np.zeros(harmonic.size)                
    MaxNPM = min(maxnpeaks, pfreq.size)
    for i in range(0, MaxNPM) :                     
        difmatrixPM = harmonic.T * np.ones(pfreq.size)
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)  
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
        PeakMag = pmag[peakloc]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
        harmonic = harmonic+f0c

    ErrorMP = np.zeros(harmonic.size)              
    MaxNMP = min(maxnpeaks, pfreq.size)
    for i in range(0, f0c.size) :                    
        nharm = np.round(pfreq[:MaxNMP]/f0c[i])
        nharm = (nharm>=1)*nharm + (nharm<1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

    Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)
    f0index = np.argmin(Error)                      
    f0 = f0c[f0index]                                

    return f0

# Function Name: noteDetect
# Input: sound signal in time domain, list of frequencies, list of notes
# Output: note of the sound signal
# Logic: taking a sample of the sound signal with blackman window, computing DFT, detecting possible peaks, using 2 way mismatch algorithm to find the note
# Example Call: note = noteDetect(sound, frequency list, note list) 
def noteDetect(tune, array, notes):
    N = 2048
    M = 1001
    t = -45
    minf0 = 50
    maxf0 = 2000
    hN = N/2
    hM = (M+1)/2
    fs = 44100
    w = get_window('blackman',M)
    if len(tune)>40000:
        y = 0.8
    elif len(tune)>20000:
        y = 0.5
    elif len(tune)>10000:
        y = 0.2
    else:
        y = 0.01
    x = tune[int(y*fs) : int(y*fs) + M]
    mX, pX = dft(x, w, N)
    ploc = peakDetect(mX, t)
    iploc, ipmag, ipphase = peakInterp(mX, pX, ploc)
    ipfreq = fs*iploc/float(N)
    f0c = np.argwhere((ipfreq>minf0)&(ipfreq<maxf0))[:,0]
    f0cf = ipfreq[f0c]
    f0 = TWM_p(ipfreq, ipmag, f0cf)
    idx = (np.abs(array-f0)).argmin()
    note = notes[idx]
    return note

# Function Name: featExt
# Input: sound signal in time domain, sampling frequency
# Output: list of features
# Logic: using librosa to extract required features, calculating mean and variance
# Example Call: feature_list = featExt(sound, fs)
def featExt(sound, fs):
    zcr = librosa.feature.zero_crossing_rate(sound, frame_length=2048, hop_length=512, center=True)  #zero crossing rate
    zcr_mean = np.mean(zcr)
    zcr_var = np.var(zcr)

    sc = librosa.feature.spectral_centroid(y=sound, sr=fs, S=None, n_fft=2048, hop_length=512, freq=None)  #spectral centroid
    sc_mean = np.mean(sc)
    sc_var = np.var(sc)

    ss = librosa.feature.spectral_bandwidth(y=sound, sr=fs, S=None, n_fft=2048, hop_length=512, freq=None, centroid=None, norm=True, p=2)  #spectral bandwidth
    ss_mean = np.mean(ss)
    ss_var = np.var(ss)

    sf = librosa.feature.spectral_flatness(y=sound, S=None, n_fft=2048, hop_length=512, amin=1e-10, power=2.0)  #spectral flatness
    sf_mean = np.mean(sf)
    sf_var = np.var(sf)

    si = librosa.feature.spectral_contrast(y=sound, sr=fs, S=None, n_fft=2048, hop_length=512, freq=None, fmin=200.0, n_bands=6, quantile=0.02, linear=False)  #spectral contrast
    si_mean = np.mean(si)
    si_var = np.var(si)

    sr = librosa.feature.spectral_rolloff(y=sound, sr=fs, S=None, n_fft=2048, hop_length=512, freq=None, roll_percent=0.85)  #spectral rolloff
    sr_mean = np.mean(sr)
    sr_var = np.var(sr)

    Mfcc = librosa.feature.mfcc(y=sound, sr=fs, S=None, n_mfcc=20, dct_type=2, norm='ortho')   #mel frequency cepstral coefficients 
    Mfcc_mean = np.mean(Mfcc)
    Mfcc_var = np.var(Mfcc)

    chr_stft = librosa.feature.chroma_stft(y=sound, sr=fs)  #chromagram
    chr_stft_mean = np.mean(chr_stft)
    chr_stft_var = np.var(chr_stft)

    chr_cqt = librosa.feature.chroma_cqt(y=sound, sr=fs)    #constant-Q chromagram
    chr_cqt_mean = np.mean(chr_cqt)
    chr_cqt_var = np.var(chr_cqt)

    chr_cens = librosa.feature.chroma_cens(y=sound, sr=fs)   #chroma energy normalized
    chr_cens_mean = np.mean(chr_cens)
    chr_cens_var = np.var(chr_cens)
    
    features = [zcr_mean, zcr_var, sc_mean, sc_var, ss_mean, ss_var, sf_mean, sf_var, si_mean, si_var, sr_mean, sr_var,Mfcc_mean, Mfcc_var,
                chr_stft_mean, chr_stft_var, chr_cqt_mean, chr_cqt_var, chr_cens_mean, chr_cens_var]
    
    return features

def Instrument_identify(audio_file):

        # defining output lists
        Instruments = []
        Detected_Notes = []
        Onsets = []
        
        #frequency list
        array = [16.35, 18.35, 20.60, 21.83, 24.50, 27.50, 30.87,
        32.70, 36.71, 41.20, 43.65, 49.00, 55.00, 61.74,
        65.41, 73.42, 82.41, 87.31, 98.00, 110.00, 123.46,
        130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94,
        261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,
        523.25, 587.33, 659.25, 698.46, 783.99, 880.00, 987.77,
        1046.50, 1174.66, 1318.51, 1396.91, 1567.98, 1760.00, 1975.53,
        2093.00, 2349.32, 2637.02, 2793.83, 3135.96, 3520.00, 3951.07,
        4186.01, 4698.63, 5274.04, 5587.65, 6271.93, 7040.00, 7902.13,
        17.32, 19.45, 23.12, 25.96, 29.14,
        34.65, 38.89, 46.25, 51.91, 58.27,
        69.30, 77.78, 92.50, 103.83, 116.54,
        138.59, 155.56, 185.00, 207.65, 233.08,
        277.18, 311.13, 369.99, 415.30, 466.16,
        554.37, 622.25, 739.99, 830.61, 932.33,
        1108.73, 1244.51, 1479.98, 1661.22, 1864.66,
        2217.46, 2489.02, 2959.96, 3322.44, 3729.31,
        4434.92, 4978.03, 5919.91, 6644.88, 7458.62]

        #corresponding notes list
        notes = ['C0','D0','E0','F0','G0','A0','B0',
        'C1','D1','E1','F1','G1','A1','B1',
        'C2','D2','E2','F2','G2','A2','B2',
        'C3','D3','E3','F3','G3','A3','B3',
        'C4','D4','E4','F4','G4','A4','B4',
        'C5','D5','E5','F5','G5','A5','B5',
        'C6','D6','E6','F6','G6','A6','B6',
        'C7','D7','E7','F7','G7','A7','B7',
        'C8','D8','E8','F8','G8','A8','B8',
        'C#0', 'D#0','F#0','G#0','A#0',
        'C#1', 'D#1','F#1','G#1','A#1',
        'C#2', 'D#2','F#2','G#2','A#2',
        'C#3', 'D#3','F#3','G#3','A#3',
        'C#4', 'D#4','F#4','G#4','A#4',
        'C#5', 'D#5','F#5','G#5','A#5',
        'C#6', 'D#6','F#6','G#6','A#6',
        'C#7', 'D#7','F#7','G#7','A#7',
        'C#8', 'D#8','F#8','G#8','A#8']

        # using pandas to read data from csv file, the csv file contrains features previously extracted from the audio files by feature extraction program
        # X stores all the features, y stores corresponding instrument in code (0 -> flute, 1 -> violin, 2 -> trumpet, 3 -> piano)
        dataset = pd.read_csv('Data_working.csv')
        X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].values
        y = dataset.iloc[:, 21].values

        # feature scaling of all features to make sure they are in the same numeric range, so that all features have equal impact in training
        sc = StandardScaler()
        X = sc.fit_transform(X)

        # calling support vector machine classifier and training the model with X and y values
        classifier = SVC(kernel = 'linear')
        classifier.fit(X, y)

        # sampling and unpacking the sound file to get sound wave in numpy array form
        file_length = audio_file.getnframes()
        sound = np.zeros(file_length)

        for i in range(file_length):
                data = audio_file.readframes(1)
                data = struct.unpack("<h", data)
                sound[i] = int(data[0])

        sound = np.divide(sound, float(2**15))

        # converting sound to mono in case it is in stereo
        #sound = librosa.core.to_mono(sound)

        # creating frames of certain window to detect and remove silence
        fs = 44100
        fd = 0.02
        frame_size = (fs * fd)
        file_length = len(sound)
        n_frames = int(round(file_length/frame_size))
        frames = []
        temp = 0
        for i in range(n_frames):
            frames.append(sound[temp:temp+int(frame_size)])
            temp = temp + int(frame_size)

        # finding sum of squares of each frame
        sq_sum_vals = [np.sum(np.square(i)) for i in frames]

        # removing silence by removing all frames that have a sum of squares less than a threshold 
        threshold = 0.002
        #threshold = 0.01
        silence = []
        for i in range(len(sq_sum_vals)):
            if sq_sum_vals[i] < threshold:
                silence.append(i)
            '''if i > 1 and i < len(sq_sum_vals)-1:
                if 5*sq_sum_vals[i] < sq_sum_vals[i+1] and 5*sq_sum_vals[i] < sq_sum_vals[i-1]:
                    silence.append(i)'''
        final_sound = sound
        
        temp = 0
        for i in range(n_frames):
            if i in silence:
                final_sound[temp:temp+int(frame_size)] = 0
            temp = temp + int(frame_size)

        # finding out starting and ending frames of each of the different tunes present in the sound file
        split_start = []
        split_end = []

        if silence[0] != 0:
                split_start.append(0)

        for i in range(len(silence)-1):
            if silence[i] != silence[i+1] - 1:
                split_start.append(silence[i])
            if silence[i] != silence[i-1] + 1:
                split_end.append(silence[i])

        if((n_frames-1)!=silence[-1]):
                split_start.append(silence[-1])
                split_end.append(n_frames - 1)

        if(0 in split_end):
            split_end.remove(0)

        for i in range(len(split_start)-1):
            if (split_start[i] > split_start[i-1] and split_start[i] < split_start[i-1]+10):
                split_start.remove(split_start[i])

        for i in range(len(split_end)-1):
            if (split_end[i] > split_end[i-1] and split_end[i] < split_end[i-1]+10):
                split_end.remove(split_end[i])

        # seperating each of the tunes using starting and ending frames
        tune = []
        temp_list = []
        for j in range(len(split_start)):
            tune.append(temp_list)
            tune[j] = final_sound[int((split_start[j])*frame_size):int((split_end[j])*frame_size)]
 
        # extracting features of each tune, detecting note and onset of each tune in a single for loop
        feature_array = []
        temp_array = []
        for j in range(len(split_start)):
            feature_array.append(temp_array)
            feature_array[j].append(temp_array)
            feature_array[j]=featExt(tune[j],fs)
            Detected_Notes.append(noteDetect(tune[j], array, notes))
            Onsets.append(split_start[j]*frame_size/fs)

        # feature scaling the extracted features in order to use for predicting the instrument
        feature_array = sc.transform(feature_array)

        # using the trained SVM classifier to predict instrument using the extracted features of each tune
        Instruments = list(classifier.predict(feature_array))

        # decoding the output to obtain corresponding instrument  
        for i in range(len(Instruments )):
            if Instruments[i] == 0:
                Instruments[i] = "Flute"
            if Instruments[i] == 1:
                Instruments[i] = "Violin"
            if Instruments[i] == 2:
                Instruments[i] = "Trumpet"
            if Instruments[i] == 3:
                Instruments[i] = "Piano"
        
        return Instruments, Detected_Notes, Onsets


############################### Main Function #############################################

if __name__ == "__main__":
        
        x = raw_input("\n\tWant to check output for all test Audio Files - Y/N: ")
                
        if x == 'Y':

                Instruments_list = []
                Detected_Notes_list = []
                Onsets_list = []
                
                file_count = len(os.listdir(path + "/Test_Audio_files"))

                for file_number in range(1, file_count):

                        file_name = path + "/Test_Audio_Files/Audio_"+str(file_number)+".wav"
                        audio_file = wave.open(file_name)

                        Instruments, Detected_Notes,Onsets = Instrument_identify(audio_file)
                        
                        Instruments_list.append(Instruments)
                        Detected_Notes_list.append(Detected_Notes)
                        Onsets_list.append(Onsets)
                print("\n\tInstruments = " + str(Instruments_list))
                print("\n\tDetected Notes = " + str(Detected_Notes_list))
                print("\n\tOnsets = " + str(Onsets_list))

