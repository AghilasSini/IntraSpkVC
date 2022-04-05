import sys
import numpy as np
import torch as T
import argarse
import glob
import os
import re

def print_stats(waveform, sample_rate=None, src=None):
	if src:
		print("-" * 10)
		print("Source:", src)
		print("-" * 10)
	if sample_rate:
		print("Sample Rate:", sample_rate)
	print("Shape:", tuple(waveform.shape))
	print("Dtype:", waveform.dtype)
	print(f" - Max:     {waveform.max().item():6.3f}")
	print(f" - Min:     {waveform.min().item():6.3f}")
	print(f" - Mean:    {waveform.mean().item():6.3f}")
	print(f" - Std Dev: {waveform.std().item():6.3f}")
	print()
	print(waveform)
	print()


def match(file_id,filelist,dirpath):
	for targetFile in filelist:
		if re.match('^{}'.format(file_id),targetFile):
			return os.path.join(dirpath,targetFile)
		



def build_arg_parser():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("filelist", help="filelist")
	parser.add_argument("pgg_dirpath", help="ppg directory container")
	parser.add_argument("mel_dirpath", help="mel directory container")
	return parser

def main():
	args=build_arg_parser().parse_args()
	pgg_dirpath=args.pgg_dirpath
	mel_dirpath=args.ppg_dirpath
	melFileList=sorted([ os.path.basename(melFile.strip()) for melFile in glob.glob(mel_dirpath)])
	ppgFileList=sorted([ os.path.basename(ppgFile.strip()) for ppgFile in glob.glob(ppg_dirpath)])
	filefull_list=[]
	with open(args.filelist,'r') as filelist:
		filefull_list=[ file_id.strip() for file_id in filelist.readlines() ]

	for file_id in  filefull_list:
		melFilePath=match(file_id,melFileList,mel_dirpath)
		ppgFilePath=match(file_id,ppgFileList,pgg_dirpath)
		if os.path.exists(melFilePath) and os.path.exists(ppgFilePath):
			# load data
			mel_data=torch.load(melFilePath)
			ppg_data=np.load(ppgFilePath)
			print("Input : ppg {}  Output: melSpec  {}".format(mel_data.shape,ppg_data.shape))
			print('Input Normal '.format(np.sum(ppg_data, axis=0)))
		else:
			sys.exit(-1)




if __name__ == '__main__':
	main()
