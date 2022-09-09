# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


# as input we have textgrid directory



# get phone occurancy





def upSampling(tgridFile,sampling_rate=22050,hop_length=221):
    # Creates a file and returns a tuple containing both the handle and the path.
    flab_path = os.path.splitext(labFile)[0]+'.flab'
    torch.zero(())

    with open(flab_path, "wb") as f:
        fid = open(labFile)
        utt_labels = fid.readlines()
        fid.close()
        for line in utt_labels:
            line = line.strip()
            temp_list = re.split('\s+', line)
            start_time = int(temp_list[0])
            end_time = int(temp_list[1])
            full_label = temp_list[2]
            
            start_samples=(start_time*1e-7)*sampling_rate
            end_samples=(end_time*1e-7)*sampling_rate

            frame_number = int(end_samples / hop_length) - int(start_samples / hop_length)
            i = 0
            while i < frame_number:
                f.write("{} {} {}\n".format(start_time + i * delta, start_time + (i + 1) * delta, full_label))
                i += 1
    return flab_path


import numpy as np




def embedding_feat(phone_to_ix,intervals,out_dir):
	n_phones=len(phone_to_ix.keys())

	lookup_tensor = torch.tensor([phone_to_ix["{}".format(phone_['phone'])]   for  phone_ in intervals['segments']  ], dtype=torch.long)
	
	onehot_frames=np.zeros((intervals['utt_end_frame'], n_phones))
	
	for embed_,seg_ in zip(F.one_hot(lookup_tensor,n_phones),intervals['segments']):
		for i in range(seg_['frame_start'],seg_['frame_end'],1):
			onehot_frames[i]=embed_.numpy()
	outFile_fpath=os.path.join(out_dir,intervals['iutt'])
	with open(outFile_fpath+'_ppg.npy','wb') as outFile:
		np.save(outFile,onehot_frames)

import TextGrid
import glob
import os

import librosa
import argparse
from  intervaltree import IntervalTree,Interval


def args_parser():
	parser = argparse.ArgumentParser(description="")
	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")
	# Add arguments
	parser.add_argument("filelist", help="The input file to be projected")
	parser.add_argument("inputdir", help="The input file to be projected")
	parser.add_argument("outdir", help="outdir to be projected")
	parser.add_argument('--label_ext',dest='label_ext',default='perslab')
	# Parsing arguments
	args = parser.parse_args()
	return args


class FileNotExistException(Exception):
	def __init__(self,file_path,message="This file does not exists"):
		self.file_path = file_path
		self.message = message
		super().__init__(self.message)
	def __str__(self):
		return f'{self.file_path} -> {self.message}'



def args_parser():
	parser = argparse.ArgumentParser(description="")
	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,
						help="increase output verbosity")
	# Add arguments
	parser.add_argument("filelist", help="The input file to be projected")
#	parser.add_argument("inputdir", help="The input file to be projected")
	parser.add_argument("outdir", help="outdir to be projected")
#	parser.add_argument('--hparams',dest='label_ext',default='perslab')
	# Parsing arguments
	args = parser.parse_args()
	return args


class FileNotExistException(Exception):
	def __init__(self,file_path,message="This file does not exists"):
		self.file_path = file_path
		self.message = message
		super().__init__(self.message)
	def __str__(self):
		return f'{self.file_path} -> {self.message}'



def get_intervals(intervals,phone_set,config_dict):
	segments=[]
	sampling_rate=config_dict['sampling_rate']
	hop_length=config_dict['sampling_rate']


	for index,interval in enumerate(intervals):
			text=interval.data
			time_start=interval.begin
			time_end=interval.end
		
			if time_start<time_end:
				if text== 'SIL' or not text:
					if index==0:
						phone='silB'
						time_end-=0.01
					elif index==len(intervals)-1:
						phone='silE'
						time_start+=0.01
					else:
						phone='pau'
				else:
					phone=text
			
				if not phone in phone_set:
					phone_set.append(phone)
				segments.append({'frame_start':int((time_start*sampling_rate)/ hop_length), 
					'frame_end':int((time_end*sampling_rate)/ hop_length), 
					'phone':phone})
			else:
				print('invalid segment {} \t {} \t {}'.format(text,time_start,time_end ))
	return segments




def main():
	args=args_parser()

	phone_set=[]
	
	utts=[]
	phone_to_ix={}
	
	sampling_rate=22050
	hop_length=222



	file_list=args.filelist
	out_dir=args.outdir


	samples= [  ]
	with open(file_list,'r') as fileList:
		for audioFilePath in fileList.readlines():
			try:
				audioFilePath=audioFilePath.strip()
				if not os.path.exists(audioFilePath):
					raise FileNotExistException(audioFilePath)
				textgridFilePath=os.path.splitext(audioFilePath)[0]+'.textgrid'
				if not os.path.exists(textgridFilePath):
					raise FileNotExistException(textgridFilePath)
				samples.append((audioFilePath,textgridFilePath))
			except FileNotExistException as err:
				print(err)

	for audioFilePath,textgridFilePath in samples:
		wav, fs =	 librosa.load(audioFilePath)
		tg=TextGrid.TextGrid()
		tg.read(textgridFilePath)
		intervals=tg['phonemes']
		utt_start_frame =int((intervals.xmin()*sampling_rate)/ hop_length)
		utt_end_frame = int((intervals.xmax()*sampling_rate)/ hop_length)
		print('number of frames {}  vs  aligned {}'.format(int(len(wav)/ hop_length),  utt_end_frame))
		items = IntervalTree([ Interval(item.xmin(),item.xmax(),item.mark()) for item in intervals  ])
		items.merge_overlaps()
		intervals=sorted(items)
		segments=get_intervals(intervals,phone_set,config_dict={'sampling_rate':sampling_rate,'hop_length':hop_length})
		utts.append({'iutt':os.path.basename(os.path.splitext(textgridFilePath)[0]),
			'segments':segments,
			'utt_start_frame':utt_start_frame,
			'utt_end_frame':utt_end_frame
		})

	

	for index,item in enumerate(sorted(phone_set)):
		phone_to_ix[item]=index

	n_samples=len(utts)
	for utt in utts:
		embedding_feat(phone_to_ix,utt,out_dir)
if __name__== '__main__':
	main()
	
	

