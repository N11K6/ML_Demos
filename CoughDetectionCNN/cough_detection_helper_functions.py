#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:56:47 2021

@author: nk
"""

import numpy as np
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub.silence import split_on_silence
#%%
def isolate_events(loadpath, savepath, min_silence_len, silence_thresh):
    '''Function to split raw audio into chunks corresponding to isolated events
    takes a specified loading path, a saving path, the minimum silence time length 
    in ms, and the threshold for silence in dB.'''
    
    sound_file = AudioSegment.from_mp3(loadpath)
    
    # split audio
    audio_chunks = split_on_silence(sound_file, 
                                    # must be silent for at least half a second
                                    min_silence_len=min_silence_len,
                                    # consider it silent if quieter than 
                                    silence_thresh=silence_thresh
                                   )
    return audio_chunks
#%%
def get_mfcc(loadpath, savepath):
    # read audio samples
    input_data, sample_rate = librosa.load(loadpath, sr=None)
    
    # ensure sample is of a reasonable length between 1 and 2 s:
    if len(input_data) >= sample_rate:
        input_data = input_data[:sample_rate]
    else:
        input_data = np.append(input_data, np.zeros(sample_rate-len(input_data)))
    
    # define figure parameters
    fig = plt.figure(figsize=[3.12,3.12])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)

    # calculate MFCCs and generate plot
    S = librosa.feature.mfcc(y=input_data, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    # store the MFCC image
    plt.savefig(savepath)
    plt.show()
