import tensorflow as tf
import librosa
import tflearn
import os
import numpy as np
width = 46 # mfcc features
height = 20
num = 20
def mfcc_generator(wave_path,PAD_WIDTH=width):
    wave,sr = librosa.load(wave_path, mono=True)
    mfccs = librosa.feature.mfcc(y=wave, sr=sr,n_mfcc=height)
    mfccs = np.pad(mfccs, ((0,0),(0, PAD_WIDTH-len(mfccs[0]))), mode='constant')
    return mfccs
script_patch=os.path.dirname(os.path.abspath( __file__ ))
sound = []
for i in range(1,num+1):
    sound.append(np.asarray(mfcc_generator("%s/dataset/0/%s.wav"%(script_patch,i))))
for i in range(1,num+1):
    sound.append(np.asarray(mfcc_generator("%s/dataset/1/%s.wav"%(script_patch,i))))  
sound = np.array(sound)
y_data = np.r_[np.c_[np.ones(num), np.zeros(num)],np.c_[np.zeros(num), np.ones(num)]]
x_test = []
for i in range(1, 11):
    x_test.append(np.asarray(mfcc_generator("%s/dataset/test/%s.wav" % (script_patch, i))))
x_test =  np.array(x_test)
y_test = np.r_[np.c_[np.ones(5), np.zeros(5)],np.c_[np.zeros(5), np.ones(5)]]

net = tflearn.input_data([None, height, width])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=500, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(sound, y_data, n_epoch=500,validation_set=(x_test,  y_test), snapshot_step = 100, show_metric=True)
  
model.save("tflearn.lstm.model")
