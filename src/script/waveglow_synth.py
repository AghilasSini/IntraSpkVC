from common.hparams import create_hparams_stage,HParamsView
from common.layers import TacotronSTFT

# usefull function
import numpy
import sys
import pickle as pkl
import soundfile
import glob
import torch
import numpy as np
import librosa

from common.audio_processing import griffin_lim

from scipy.io.wavfile import write



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


import sys
import numpy as np
import torch
import os
import argparse

from common.hparams import create_hparams

from common.layers import TacotronSTFT

from scipy.io.wavfile import write

# Waveglow
from common.utils import waveglow_audio, get_inference, load_waveglow_model
from waveglow.denoiser import Denoiser


from scipy.io import wavfile
from script.train_ppg2mel import load_model
from common.utils import get_inference
from common.valid_file import FileNotExistException
import json


def infer(checkpoint_path, waveglow_path, ppg_files, out_dirname,hparams):
	 
		tacotron_model = load_model(hparams)
		tacotron_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
		_ = tacotron_model.cuda().eval()
		
		
		
		
		
		
		taco_stft = TacotronSTFT(
		hparams.filter_length, hparams.hop_length, hparams.win_length,
		hparams.n_acoustic_feat_dims, hparams.sampling_rate,
		hparams.mel_fmin, hparams.mel_fmax)
		is_clip = False
		
		
		#waveglow
		waveglow_for_denoiser = torch.load(waveglow_path)['model']
		waveglow_for_denoiser.cuda()
		denoiser_mode = 'zeros'
		denoiser = Denoiser(waveglow_for_denoiser, mode=denoiser_mode)

		waveglow_model = load_waveglow_model(waveglow_path)

		#waveglow=waveglow.to('cpu')
		#.half()
		#for k in waveglow.convinv:
		#    k.float()
		#denoiser = Denoiser(waveglow)
		# End of parameters

		
		waveglow_sigma= 1#0.66
		denoiser_strength=0.005
		for i, ppg_fpath in enumerate(ppg_files):
			ppg_data = np.load(ppg_fpath)
			gen_mel_filename=os.path.basename(ppg_fpath).split('.')[0]
			mel_outputs_postnet = get_inference(ppg_data, tacotron_model, is_clip)
			
			ac_wav = waveglow_audio(mel_outputs_postnet, waveglow_model,
                                waveglow_sigma, True)
			audio = denoiser(ac_wav, strength=denoiser_strength)[:, 0].cpu().numpy().T

			
			
			#audio = audio_denoised.cpu().numpy()
			#audio = audio.astype('int16')
			audio_path = os.path.join(out_dirname, "{}_wg-synth.wav".format(gen_mel_filename))
			write(audio_path, hparams.sampling_rate, audio)
			#print("saved in {}".format(audio_path))
			

if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument('-p', '--ppg', type=str,help='ppg to infer')
		parser.add_argument('-w', '--waveglow', type=str,help='waveglow models')
		parser.add_argument('-c', '--checkpoint', type=str,help='checkpoint path')
		parser.add_argument('--hparams', type=str,help='checkpoint path')
		parser.add_argument('-o', '--out_dirname', type=str, help='output filename', default='sample')
		args = parser.parse_args()
		# load hparams
		with open(args.hparams,'r') as hpFile:
			hparams = HParamsView(json.load(hpFile))
		# load ppg files
		ppg_files=[]
		with open(args.ppg,'r') as ppgFileList:
			for ppg_fpath in ppgFileList.readlines():
				try:
					ppg_fpath=ppg_fpath.strip()
					if not os.path.exists(ppg_fpath):
						raise FileNotExistException(ppg_fpath)
					ppg_files.append(ppg_fpath)
				except FileNotExistException as err:
					print(err)
					
		
		infer(args.checkpoint, args.waveglow, ppg_files, args.out_dirname,hparams)



