# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np
import librosa as lb
from tensorflow import keras
#%%
PATH_TO_AUDIO = 'path/to/an/audio/file.wav'
PATH_TO_MODEL = 'path/to/the/trained/model.h5'
#%%
def segment_audio(path_to_audio):
    '''
    Segments audio file from path into 1s long segments, using a hop size of
    0.5s, and returns them in a list
    '''
    audio, sr = lb.load(path_to_audio)
    len_audio = audio.shape[0]
    n_window = sr
    
    chunks = []

    if len_audio <= n_window:
        chunk = np.append(audio, np.zeros(sr-len_audio))
        chunks.append(chunk)    
    else:
        
        hop = sr//2
        n_frames = (len_audio-n_window) // hop

        ind = 0

        for i in range(n_frames):
            chunk = audio[ind:ind+n_window]
            chunks.append(chunk)
            ind += hop
    
    return chunks
#%%
def get_mfcc_features(chunks):
    '''
    Extracts MFCC features from audio chunks in input list, stores them
    in a 3D numpy array.
    '''
    mfccs = []

    for chunk in chunks:
        mfcc = lb.feature.mfcc(chunk)
        mfccs.append(mfcc)
    
    mfccs = np.array(mfccs)
    
    return mfccs
#%%
def make_prediction(path_to_model, mfccs):
    '''
    Uses trained model to make predictions on the extracted features.
    '''
    # Load model
    model = keras.models.load_model(path_to_model)
    # Expand feature dimensions
    features = np.expand_dims(mfccs, axis=-1)
    # Make predictions
    predictions = model.predict(features)
    
    return predictions
#%%
def main():
    
    CHUNKS = segment_audio(PATH_TO_AUDIO)
    MFCCS = get_mfcc_features(CHUNKS)
    PREDICTIONS = make_prediction(PATH_TO_MODEL, MFCCS)
    
    print(np.round(PREDICTIONS,0))
    
    return PREDICTIONS
#%%
if __name__ == "__main__":
    main()