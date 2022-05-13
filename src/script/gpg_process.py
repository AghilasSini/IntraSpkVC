import sys
import os
import codecs
import argparse

import multiprocessing as mp
from tqdm import tqdm


# ----------------------------- #
# 
# 

import soundfile as sf
import torch
from torch import nn
#from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa






def compute_gpg(args):
    sample, model_name,out_dir_path=args
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    speech, sample_rate = sf.read(sample)
    if sample_rate!=16000:
        speech = librosa.resample(speech, sample_rate, 16000)
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    logits = model(input_values).logits
    gpg = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()
    output_full_path=os.path.join(out_dir_path,"{}_gpg.npy".format(os.path.basename(sample).split('.')[0]))
    np.save(output_full_path, gpg)

    return gpg



def build_arg_parser():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("in_file_list", help="audio file")
	parser.add_argument("out_path_dir", help="audio file")
	
	return parser



def main():


	args=build_arg_parser().parse_args()

	in_file_list=args.in_file_list
	out_path_dir=args.out_path_dir


	model_name="facebook/wav2vec2-large-xlsr-53-french"
	
	
	with codecs.open(in_file_list,'r','utf-8') as dataFileList:
		dataset=[sample.strip() for sample in  dataFileList.readlines() if os.path.exists(sample.strip())]

	n_samples=len(dataset)
	with mp.Pool(mp.cpu_count()) as p:
			gpg_dataset=list(
				tqdm(
					p.imap(
						compute_gpg,
						zip(
							dataset,
							[model_name] * n_samples,
							[out_path_dir] * n_samples,
							
						),
					),
					total=n_samples,
				)
			)
	


if __name__ == '__main__':
	main()



