# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import matplotlib.pyplot as plt



meta = pd.read_csv('../data/ESC-50/meta/esc50.csv')

meta.head()

plt.figure(figsize=(16, 9))
meta['category'].value_counts().plot.bar()



N = meta.shape[0]
N



data = [None] * N

file_root = '../data/ESC-50/audio/'
file_list = meta['filename'].values
file_list

SR = 44100
n_fft = 2048
win_length = 2048
hop_length = 1024
n_mels = 128

for i in tqdm(range(N)):
    path = file_root + file_list[i]
    y, sr = librosa.load(path, sr=SR, mono=False)
    S = librosa.feature.melspectrogram(y, sr=SR, n_mels = n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    S = librosa.core.power_to_db(S)
    data[i] = S

data = np.array(data) 
data = data[:, :, :, np.newaxis]

cats = ['dog', 'crying_baby', 'door_wood_knock', 'helicopter']


def plot_melspectrogram(c):
    idx = (meta['category'] == c).idxmax()
    S = data[idx, :, :, 0]
    plt.imshow(np.flip(S, axis=0), cmap='magma')
    plt.xlabel('Time')
    plt.ylabel('mel')
    plt.title(c)
    plt.savefig('../png/'+ c + '.png')
    plt.close()


for c in cats:
    plot_melspectrogram(c)

index = meta.index.values

for i in range(5):
    j = i+1
    np.save('../data/fold' + str(i) + '.npy', index[(meta['fold'] ==  j)])



print('mean {}'.format(data.mean()))
print('std {}'.format(data.std()))



np.save('../data/melspectrogram.npy', data)


