import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

import io
import os
import math
import tarfile
import multiprocessing

import scipy
import librosa
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time

# from IPython.display import Audio, display
import numpy
import sys
import pickle as pkl
import soundfile
import glob

def plot_spectrogram(spec,fname, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(spec, origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)

  plt.savefig(fname)

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  # waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  fname='test_audio'
  plt.savefig(fname)
  # plt.show(block=False)



test_dir_fpath=sys.argv[1]


hop_length=221
win_length=None
n_fft=1024
MAX_WAV_VALUE = 32768.0

# melspec_librosa = librosa.feature.melspectrogram(
#     waveform.numpy()[0],
#     sr=sample_rate,
#     n_fft=n_fft,
#     hop_length=hop_length,
#     win_length=win_length,
#     center=True,
#     pad_mode="reflect",
#     power=2.0,
#     n_mels=n_mels,
#     norm='slaney',
#     htk=True,
# )



for mel_fl in glob.glob('{}/*.pt'.format(test_dir_fpath)):
  # print(fl)
  spect=torch.load(mel_fl)
  print(spect.shape)
  saved_file=os.path.splitext(mel_fl)[0]
  # if os.path.splitext(saved_file)[1]!='wav':
  #   saved_file=saved_file+'.wav'
  fname=saved_file.replace('.wav','.png')
  print(fname)
  # # print(spect.shape)
  mel=spect.detach().numpy()
  plot_spectrogram(mel,fname, title="MelSpectrogram - torchaudio", ylabel='mel freq')
  # mel=spect.detach().numpy()
  # wav_data_2 = librosa.feature.inverse.mel_to_audio(mel, sr=22050, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
  # soundfile.write(saved_file, wav_data_2, 22050)


  




# # plot_waveform(wav_data_2,22050)

# 