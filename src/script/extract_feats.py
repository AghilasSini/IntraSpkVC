# experimental

# extract MFCC and Ivectors for FFR0009/FFR0012/MFR0015 


# for given source and target speaker, compute for both speaker the features (mfcc and ivector) used for extracting ppg.
# args @source speaker audio files directory
#	   @target speaker audio files directory
# output
#	   @out_dir (speaker_id/ivectors/mfcc) 





import os
import sys


import pandas as pd
import numpy as np



# Argument
import argparse

# Debug
import traceback
import time

import codecs

from tqdm import tqdm
import multiprocessing as mp

import time


from common import layers
from common.hparams import HParamsView

import torch

import pickle as pkl


sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

from glob import glob 
__module_dir = os.path.dirname(__file__)
BG_COLOR = '#FFFFFF'
ccolor = '#555555'

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..', '..','data/synpaflex_corpus/v0.1')



def args_parser():
	parser = argparse.ArgumentParser(description="")
	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")
	# Add arguments
	parser.add_argument("wav_file_list",type=str, help="The input files list")
	parser.add_argument("hparams",type=str, help="hparams")
	parser.add_argument("output_feat_dir",type=str, help="Output directory acoustic features")
	args = parser.parse_args()
	return args


from scipy.io import wavfile


def extract_acoustic_feat(feat):

	row,hparams,output_feat_dir=feat


	stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_acoustic_feat_dims, hparams.sampling_rate,
            hparams.mel_fmin, hparams.mel_fmax)


	fs, wav = wavfile.read(row['audio_fpath'])
	audio = torch.FloatTensor(wav.astype(np.float32))
	# print(data)
	

	if fs != stft.sampling_rate:
		raise ValueError("{} SR doesn't match target {} SR".format(
			fs, stft.sampling_rate))
	audio_norm = audio / hparams.max_wav_value
	audio_norm = audio_norm.unsqueeze(0)
	audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
	# (1, n_mel_channels, T)
	acoustic_feats = stft.mel_spectrogram(audio_norm)
	# (n_mel_channels, T)
	acoustic_feats = torch.squeeze(acoustic_feats, 0)
	# (T, n_mel_channels)
	acoustic_feats = acoustic_feats.transpose(0, 1)
	out_melspec_pkl=os.path.join(output_feat_dir,"{}_mel.pkl".format(os.path.basename(row['audio_fpath']).split('.')[0]))
	with open(out_melspec_pkl,'wb') as f:
		pkl.dump(acoustic_feats, f)


	# out_melspec_pt=os.path.join(output_feat_dir,"{}_mel.pt".format(os.path.basename(row['audio_fpath'])))

	

import json
import codecs
			
def main():
	
	args=args_parser()
	wav_file_list=args.wav_file_list
	output_feat_dir=args.output_feat_dir

	if not os.path.exists(output_feat_dir):
		print(output_feat_dir)
		os.makedirs(output_feat_dir)


	wav_file_list=pd.read_csv(wav_file_list,sep=' ',names=['audio_id','audio_fpath'])
	samples=wav_file_list.to_dict('records')
	n_samples=len(samples)

	hparams_filename=args.hparams
	with codecs.open(hparams_filename) as hpFile:
		hparams=HParamsView(json.load(hpFile))



	with mp.Pool(mp.cpu_count()-10) as p:
            result=list(
                tqdm(
                    p.imap(
                        extract_acoustic_feat,
                        zip(
                            samples,  
                            [hparams]*n_samples,
                            [output_feat_dir]*n_samples
                        
                        ),
                    ),
                    total=n_samples,
                )
            )   






if __name__ == '__main__':
	main()



