import tensorflow as tf
import librosa
import tflearn
import os
import numpy as np

width = 80 # mfcc features
height = 20

script_patch=os.path.dirname(os.path.abspath( __file__ ))
net = tflearn.input_data([None, height, width])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=500, loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.load("tflearn.lstm.model")

def mfcc_generator(wave_path,PAD_WIDTH=width):
    wave,sr = librosa.load(wave_path, mono=True)
    mfccs = librosa.feature.mfcc(y=wave, sr=sr,n_mfcc=height)
    mfccs = np.pad(mfccs, ((0,0),(0, PAD_WIDTH-len(mfccs[0]))), mode='constant')
    return mfccs

sounds = []
num = 4
for i in range(1, num + 1):
    sounds.append(np.asarray(mfcc_generator("%s/dataset/try/%s.wav"%(script_patch,i))))
              
sounds = np.array(sounds)

abc =  np.round(model.predict(sounds))

print(abc)    
