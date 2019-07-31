#This program uses Librosa to extract features from all the labelled audio files played by different instruments

#Import Modules
import numpy as np
import pandas as pd
import librosa
import scipy
import wave
import struct
import os

# Function Name: feat_ext
# Input: sound file, sampling frequency and instrument code number
# Output: list of all extracted features
# Logic: using librosa to extract the required features, calculating mean and variance using numpy and storing them in a list
# Example Call: features = feat_ext(sound, fs, ins)
def feat_ext(sound, fs, ins):
    zcr = librosa.feature.zero_crossing_rate(sound, frame_length=2048, hop_length=512, center=True)
    zcr_mean = np.mean(zcr)
    zcr_var = np.var(zcr)

    sc = librosa.feature.spectral_centroid(y=sound, sr=fs, S=None, n_fft=2048, hop_length=512, freq=None)
    sc_mean = np.mean(sc)
    sc_var = np.var(sc)

    ss = librosa.feature.spectral_bandwidth(y=sound, sr=fs, S=None, n_fft=2048, hop_length=512, freq=None, centroid=None, norm=True, p=2)
    ss_mean = np.mean(ss)
    ss_var = np.var(ss)

    sf = librosa.feature.spectral_flatness(y=sound, S=None, n_fft=2048, hop_length=512, amin=1e-10, power=2.0)
    sf_mean = np.mean(sf)
    sf_var = np.var(sf)

    si = librosa.feature.spectral_contrast(y=sound, sr=fs, S=None, n_fft=2048, hop_length=512, freq=None, fmin=200.0, n_bands=6, quantile=0.02, linear=False)
    si_mean = np.mean(si)
    si_var = np.var(si)

    sr = librosa.feature.spectral_rolloff(y=sound, sr=fs, S=None, n_fft=2048, hop_length=512, freq=None, roll_percent=0.85)
    sr_mean = np.mean(sr)
    sr_var = np.var(sr)

    Mcff = librosa.feature.mfcc(y=sound, sr=fs, S=None, n_mfcc=20, dct_type=2, norm='ortho')
    Mcff_mean = np.mean(Mcff)
    Mcff_var = np.var(Mcff)

    chr_stft = librosa.feature.chroma_stft(y=sound, sr=fs)
    chr_stft_mean = np.mean(chr_stft)
    chr_stft_var = np.var(chr_stft)

    chr_cqt = librosa.feature.chroma_cqt(y=sound, sr=fs)
    chr_cqt_mean = np.mean(chr_cqt)
    chr_cqt_var = np.var(chr_cqt)

    chr_cens = librosa.feature.chroma_cens(y=sound, sr=fs)
    chr_cens_mean = np.mean(chr_cens)
    chr_cens_var = np.var(chr_cens)

    mel_spec = librosa.feature.melspectrogram(y=sound, sr=fs)
    mel_mean = np.mean(mel_spec)
    mel_var = np.var(mel_spec)

    rms = librosa.feature.rmse(y=sound)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    
    features = [zcr_mean, zcr_var, sc_mean, sc_var, ss_mean, ss_var, sf_mean, sf_var, si_mean, si_var, sr_mean, sr_var, Mcff_mean, Mcff_var,
                chr_stft_mean, chr_stft_var, chr_cqt_mean, chr_cqt_var, chr_cens_mean, chr_cens_var, ins]

    return features

#feature arrays for each instrument
feature_array_flute = []
feature_array_violin = []
feature_array_trumpet = []
feature_array_piano = []

#number of files i.e sound samples for each instrument
file_count = 28

#instrument code 0 for flute
ins = 0

#opens each audio file present in Flute folder, unpacks them and extracts their features to store in the feature array
for file_number in range(1, file_count+1):
    file_name = "Dataset/Flute/Audio ("+str(file_number)+").wav"
    audio_file = wave.open(file_name)
    file_length = audio_file.getnframes()
    sound = np.zeros(file_length)

    for i in range(file_length):
            data = audio_file.readframes(1)
            data = struct.unpack("<h", data)
            sound[i] = int(data[0])

    sound = np.divide(sound, float(2**15))
    fs = 44100
    tune = librosa.core.to_mono(sound)
    fd = 0.01
    temp_array = []
    feature_array_flute.append(temp_array)
    feature_array_flute[file_number-1]=feat_ext(tune,fs,ins)

#instrument code 1 for violin
ins = 1

#opens each audio file present in Violin folder, unpacks them and extracts their features to store in the feature array
for file_number in range(1, file_count+1):
    file_name = "Dataset/Violin/Audio ("+str(file_number)+").wav"
    audio_file = wave.open(file_name)
    file_length = audio_file.getnframes()
    sound = np.zeros(file_length)

    for i in range(file_length):
            data = audio_file.readframes(1)
            data = struct.unpack("<h", data)
            sound[i] = int(data[0])

    sound = np.divide(sound, float(2**15))
    fs = 44100
    tune = librosa.core.to_mono(sound)
    fd = 0.01
    frame_size = (fs * fd)
    file_length = len(sound)
    n_frames = int(round(file_length/frame_size))
    temp_array = []
    feature_array_violin.append(temp_array)
    feature_array_violin[file_number-1]=feat_ext(tune,fs,ins)

#instrument code 2 for trumpet
ins = 2

#opens each audio file present in Trumpet folder, unpacks them and extracts their features to store in the feature array
for file_number in range(1, file_count+1):
    file_name = "Dataset/Trumpet/Audio ("+str(file_number)+").wav"
    audio_file = wave.open(file_name)
    file_length = audio_file.getnframes()
    sound = np.zeros(file_length)

    for i in range(file_length):
            data = audio_file.readframes(1)
            data = struct.unpack("<h", data)
            sound[i] = int(data[0])

    sound = np.divide(sound, float(2**15))
    fs = 44100
    tune = librosa.core.to_mono(sound)
    fd = 0.01
    frame_size = (fs * fd)
    file_length = len(sound)
    n_frames = int(round(file_length/frame_size))
    temp_array = []
    feature_array_trumpet.append(temp_array)
    feature_array_trumpet[file_number-1]=feat_ext(tune,fs,ins)

#instrument code 3 for piano
ins = 3

#opens each audio file present in Piano folder, unpacks them and extracts their features to store in the feature array
for file_number in range(1, file_count+1):
    file_name = "Dataset/Piano/Audio ("+str(file_number)+").wav"
    audio_file = wave.open(file_name)
    file_length = audio_file.getnframes()
    sound = np.zeros(file_length)

    for i in range(file_length):
            data = audio_file.readframes(1)
            data = struct.unpack("<h", data)
            sound[i] = int(data[0])

    sound = np.divide(sound, float(2**15))
    fs = 44100
    tune = librosa.core.to_mono(sound)
    fd = 0.01
    frame_size = (fs * fd)
    file_length = len(sound)
    n_frames = int(round(file_length/frame_size))
    
    temp_array = []
    feature_array_piano.append(temp_array)
    feature_array_piano[file_number-1]=feat_ext(tune,fs,ins)

#make a big list that contains all the features
feature_array = feature_array_flute + feature_array_violin + feature_array_trumpet + feature_array_piano

#convert the list into a pandas dataframe
x=pd.DataFrame(feature_array)

#write the data to a csv file using pandas
x.to_csv('Data_working.csv')
print("Done!")



