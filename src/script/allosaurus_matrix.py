#!/usr/bin/env python
# coding: utf-8
# author Aghilas SINI

from allosaurus.app import read_recognizer
import re
from argparse import Namespace
import numpy as np

import pandas as pd
import os
import argparse


from allosaurus.bin import  list_phone

from tqdm import tqdm
import multiprocessing as mp

import roots


from allosaurus.lm.inventory import Inventory
from allosaurus.model import get_model_path


import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from pocketsphinx import Pocketsphinx, Jsgf

import codecs

import GenerateJSpeechGrammar as gjsgf

import librosa
import math
import glob


import soundfile as sf
### forced alignment methods utils

def forcing_audio_properties(audio_file,audio_file_name):
	data, samplerate = sf.read(audio_file)
	data = data.T
	data = librosa.to_mono(data)
	data_16k = librosa.resample(data, samplerate, 16000)
	#overwrite wavfile
	sf.write(audio_file_name, data_16k,16000,subtype='PCM_16')



def get_segment(ps,jsgf_object,rule_name,wavfile):
	try:
		rule = jsgf_object.get_rule('utt.{}'.format(rule_name))
		fsg = jsgf_object.build_fsg(rule, ps.get_logmath(), 7.5)
		ps.set_fsg('{}'.format(rule_name), fsg)
		ps.set_search('{}'.format(rule_name))
		

		ps.decode(

		audio_file=wavfile,
			buffer_size=2048,
			no_search=False,
			full_utt=False,
			)
		return [ {'token':seg_.word,'start_frame':"{:.3f}".format(seg_.start_frame),'end_frame':"{:.3f}".format(seg_.end_frame), 'dscore': "{:.3f}".format( math.log(float(seg_.end_frame)-float(seg_.start_frame)+1)) ,"ascore":"{:.3f}".format(-math.log(1-float(seg_.ascore)))} for seg_ in ps.seg()]
	except RuntimeError as err:
		if err == 'RuntimeError: Decoder_set_fsg returned -1':
			return []
		else:
			return []

def flatten(l): return [item for sublist in l for item in sublist]



def set_phone_dict(phonemes,filename_output,language):
	update_dict={}
	
	
	for phone in flatten([ph.split() for ph in phonemes]):
		if not phone in update_dict.keys():
			update_dict[phone]=re.sub("SS","S",phone.upper()) if  language=='en' or language=='en-us' else phone.lower()
		else:
			pass

	
	with codecs.open(filename_output,'w','utf-8') as dictFile:
		for item in update_dict.keys():
			dictFile.write('{} {}\n'.format(item,update_dict[item]))
		dictFile.write('sil SIL')
	print('the new dictionary have been generated')

def is_file_empty(file_path):
	""" Check if file is empty by confirming if its size is 0 bytes"""
	# Check if file exist and it is empty
	isEmpty=True
	with codecs.open(file_path,'r','utf-8') as isEmptyFile:
		if len(isEmptyFile.readlines()) <1:
			isEmpty=False



	return  isEmpty
	


def is_rule_matching(grammar, speech):
	return grammar.find_matching_rules(speech)


class EmptyFileException(Exception):
	def __init__(self,file_path,message="This file is empty"):
		self.file_path = file_path
		self.message = message
		super().__init__(self.message)
	
	def __str__(self):
		return f'{self.file_path} -> {self.message}'


	
class NoRulesMatchedException(Exception):
	def __init__(self,rule_name,message="No rules matched"):
		self.rule_name=rule_name
		self.message=message
		super().__init__(self.message)
	
	def __str__(self):
		return f'{self.rule_name} -> {self.message}'









### Framing methods utilities







def stride_trick(a, stride_length, stride_step):
		 """
		 apply framing using the stride trick from numpy.

		 Args:
				 a (array) : signal array.
				 stride_length (int) : length of the stride.
				 stride_step (int) : stride step.

		 Returns:
				 blocked/framed array.
		 """
		 nrows = ((a.size - stride_length) // stride_step) + 1
		 n = a.strides[0]
		 return np.lib.stride_tricks.as_strided(a,
																						shape=(nrows, stride_length),
																						strides=(stride_step*n, n))


def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
		 """
		 transform a signal into a series of overlapping frames (=Frame blocking).

		 Args:
				 sig     (array) : a mono audio signal (Nx1) from which to compute features.
				 fs        (int) : the sampling frequency of the signal we are working with.
													 Default is 16000.
				 win_len (float) : window length in sec.
													 Default is 0.025.
				 win_hop (float) : step between successive windows in sec.
													 Default is 0.01.

		 Returns:
				 array of frames.
				 frame length.

		 Notes:
		 ------
				 Uses the stride trick to accelerate the processing.
		 """
		 # run checks and assertions
		 if win_len < win_hop: print("ParameterError: win_len must be larger than win_hop.")

		 # compute frame length and frame step (convert from seconds to samples)
		 frame_length = win_len * fs
		 frame_step = win_hop * fs
		 signal_length = len(sig)
		 frames_overlap = frame_length - frame_step

		 # compute number of frames and left sample in order to pad if needed to make
		 # sure all frames have equal number of samples  without truncating any samples
		 # from the original signal
		 rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
		 pad_signal = np.append(sig, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))

		 # apply stride trick
		 frames = stride_trick(pad_signal, int(frame_length), int(frame_step))
		 return frames, frame_length




def forced_alignment(audio_file,phonemes,language,acmod,data_dir):
	filename,ext=os.path.basename(audio_file).split('.')
	wavfile=os.path.join(data_dir,filename+'.wav')
	forcing_audio_properties(audio_file,wavfile)

	jsgfile = gjsgf.GenerateJSpeechGrammar(phonemes,filename,data_dir)
	jsgfile.utterance_grammar_generation(language=language)

	jsgfile_path=jsgfile.get_grammar_file_path()

	new_dict=os.path.join(data_dir,'{}.dict'.format(filename))
	set_phone_dict(phonemes,new_dict,language)
	try:
		if not is_file_empty(new_dict):
			raise EmptyFileException(new_dict)
		if not is_file_empty(jsgfile_path):
			raise EmptyFileException(jsgfile_path)
		if not os.path.exists(wavfile):
			raise EmptyFileException(wavfile)

		config = {
			'hmm': acmod,
			'lm':False ,#os.path.join(model_path, 'en-us.lm.bin'),
			'dict':new_dict, # os.path.join(get_model_path, 'cmudict-en-us.dict')
			"beam": 1e-57,
			"wbeam": 1e-56,
			"maxhmmpf": -1,
			"fsgusefiller": False
		}
		
		ps = Pocketsphinx(**config)
		jsgf = Jsgf(jsgfile_path)

		phones_scores=get_segment(ps,jsgf,'phonemes',wavfile)


		# [print('word {} dscore {} ascore {}'.format(seg_['token'],seg_['dscore'],seg_['ascore'])) for seg_ in phones_scores]
		return [seg_ for seg_ in phones_scores if seg_['token']!='(NULL)']
	except EmptyFileException as exp:
		print(exp)
	except NoRulesMatchedException as noRule:
		print(noRule)






def build_arg_parser():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument("filelist", help="audio file list")
	parser.add_argument("outdir_path", help="text file")
	parser.add_argument('--language',dest='lang',default='fra')
	parser.add_argument('--model_name',dest='model_name',default='fra2105cv')
	parser.add_argument('--acmod_path',dest='acmod_path',default='/mnt/mydrive/home/aghilas/Workspace/Experiments/Nadine/allopreval/data/models/cmusphinx-fr-5.2')

	return parser


def run(feat):
	audiofile,config=feat
	# model=config['model']
	allosaurus_language=config['lang']
	phone_list=config['phone_list']
	outdir_path=config['outdir']
	phoneset_map=config['phoneset_map']
	acmod=config['acmod_path']
	# print(audiofile,allosaurus_language,phone_list,outdir_path)

	allosaurus_config = Namespace(model='fra2105cv', device_id=-1,timestamp=True, lang=allosaurus_language, approximate=False, prior=None)
	model = read_recognizer(allosaurus_config)


	result=model.recognize(audiofile,timestamp=True, lang_id=allosaurus_language, topk=len(phone_list))
	dict_data={}
	candates_T=result.split('\n')
	T=len(candates_T)
	# initialize dictionary
	updated_phone_list=phone_list+['<blk>']


	df = pd.DataFrame(columns=updated_phone_list, index=range(T))
	df = df.fillna(float(0.)) 
	# print(df.head())
	for phone in updated_phone_list:
		dict_data[phone]=float(0.0)
	# # fill the probabilities matrix T X M (M: Number of phones, T: Number of frames)
	waveform, fs = torchaudio.load(audiofile)
	phones_ipa_seq=[]
	prev_phone_end=0
	for t_idx,t in enumerate(candates_T):
		feat_=t.split(' ')
		phones_canditates=feat_[2:]
		i=0
		candidate_t=dict_data

		while i <len(phones_canditates):

			phone_transcription=phones_canditates[i].strip()

			phone_probability=float(re.sub('\(|\)','',phones_canditates[i+1]))
			candidate_t[phone_transcription]=phone_probability
			i+=2
			# df.
	
		df.iloc[t_idx]=pd.Series(candidate_t)
		phone_ipa=feat_[2].strip()
		start_t=float(feat_[0])
		duration=float(feat_[1])
		cur_phone_end=start_t+duration
		if prev_phone_end<start_t:
			sig_seg=waveform[0,int(prev_phone_end*fs):int(start_t*fs)+1]

			phones_ipa_seq.append({'label':'sil',
				 'start_t':prev_phone_end,
				 'end_t':start_t,
				 'duration':start_t-prev_phone_end,
				 'frames': len(framing(sig_seg,fs=fs)[0])
		})
		sig_seg=waveform[0,int(start_t*fs):int(cur_phone_end*fs)+1]

		phones_ipa_seq.append({'label':phone_ipa,
						'start_t':start_t,
						'end_t':cur_phone_end,
						'duration':len(sig_seg)/fs,
						'frames': len(framing(sig_seg,fs=fs)[0])
			})
		prev_phone_end=cur_phone_end

	sig_seg=waveform[0,int(cur_phone_end*fs):waveform.shape[1]+1]
	if cur_phone_end< waveform.shape[1]/fs:
		 phones_ipa_seq.append({'label':'sil',
				 'start_t':cur_phone_end,
				 'end_t':waveform.shape[1]/fs,
				 'duration':(waveform.shape[1]/fs)-prev_phone_end,
				 'frames': len(framing(sig_seg,fs=fs)[0])
		})
	# print("{}".format())
	npy_data=df.to_numpy()
	phonesIpa_SeqWithSil=[  item['label'] for item in phones_ipa_seq]



	# print(f" - Max:     {npy_data.max().item():6.3f}")
	# print(f" - Min:     {npy_data.min().item():6.3f}")
	# print(f" - Mean:    {npy_data.mean().item():6.3f}")
	# print(f" - Std Dev: {npy_data.std().item():6.3f}")
	# print(" - Valid row prob. {} ".format(np.sum(npy_data,axis=1)))
	# print("arg max")

	argmax_df=df.idxmax(axis="columns").to_dict()
	phonesIpa_sequence=[v  for _,v in argmax_df.items() ]

	if len(phonesIpa_sequence)>0:
		phoneLia_sequence=convert_Ipa2Lia(phonesIpa_sequence,allosaurus_language,phoneset_map)
		
		segments_=forced_alignment(audiofile,phoneLia_sequence,allosaurus_language,acmod,outdir_path)	
		print(waveform.shape[1]/fs)
		
		data_align=["{} {} {}".format(seg_['token'],seg_['start_frame'],seg_['end_frame']) for seg_ in segments_ ]
		if len(data_align)==0:
			print('take the default feature  filename {}'.format(os.path.basename(audiofile)))



		else:
			print('align the above feature with ')
			rec_len=df.shape[0]
			ali_len=len([ seg_ for seg_ in segments_ if seg_['token']!='sil' ])
			if rec_len==ali_len:
				
				alp_full=fill_alp_tensor(segments_,df)

				print("new shape {}".format(alp_full.shape))
				
				print('convert to numpy tensor')
				alp_full_npy=alp_full.to_numpy()
				print(np.sum(alp_full_npy,axis=1))
				out_file=os.path.join(outdir_path, os.path.splitext(os.path.basename(audiofile))[0]+'_alp')
				# save to numpy data 
				np.save(out_file,alp_full_npy)



			else:
				print('something goes wrong with these two sequence')
	else: 
		pass



def fill_alp_tensor(ali_segs,rec_cands_segs):
	
	nframes= int(float(ali_segs[-1]['end_frame']))
	# 
	df_alp_full = pd.DataFrame(columns=rec_cands_segs.columns, index=range(nframes))
	df_alp_full = df_alp_full.fillna(float(0.))
	sil_temp={}
	for phone in rec_cands_segs.columns:
		if phone=='<blk>':
			sil_temp[phone]=float(1.0)
		else:
			sil_temp[phone]=float(0.0)

	
	beg_frame=0
	iphone=0

	for ali_seg in ali_segs:
	
		if ali_seg['token']=='sil':
			end_frame= int(float(ali_seg['end_frame']))
			for iframe in range(beg_frame,end_frame):
				df_alp_full.iloc[iframe]=pd.Series(sil_temp)
			beg_frame=end_frame
		else:
			end_frame= int(float(ali_seg['end_frame']))
			for iframe in range(beg_frame,end_frame):
				df_alp_full.iloc[iframe]=rec_cands_segs.iloc[iphone]
			beg_frame=end_frame
			iphone+=1
	return df_alp_full




	# forced_alignment()

	# for align with pocketsphinx firt convert following ipa phone to liaphon phone dict

	# sorted_df={k: v for k, v in sorted(argmax_df.items(), key=lambda item: item[1])}
	# print([k for k,_ in sorted_df.items()])

	
	# print(f" - Std Dev: {np.sum(npy_data,axis=0):6.3f}")
	# # define output filename
	# out_file=os.path.join(outdir_path, os.path.splitext(os.path.basename(audiofile))[0]+'_alp')
	# # save to numpy data 
	# np.save(out_file,npy_data)




import unicodedata
def strip_accents(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def convert_Ipa2Lia(phones_sequence,lang,phoneset_map):
	if lang=='fra':
		phoneset_alpha_mapping=pd.read_csv(phoneset_map)
	else:
		print('for now this language is not supported')
	liaphon_sample=[]
	for phone_ipa in phones_sequence:
		if phone_ipa in phoneset_alpha_mapping['IPA'].to_list():
			# print("no special diacritic is detected in this phone  {} Valid Phone ".format(phone_ipa))
			pass
		else:
			# print("this phone is missing or has a particular diacritic {}   ".format(phone_ipa))
			phone_ipa=strip_accents(phone_ipa)
			if phone_ipa in phoneset_alpha_mapping['IPA'].to_list():
				# print("after cleaning up the  diacritics this phone {}  become  Valid Phone ".format(phone_ipa))
				pass
			else:
				# print("this phone is not a valid french phone {} Invalid Phone ".format(phone_ipa))
				phone_ipa='sil'
		liaphon_sample.append(phoneset_alpha_mapping.loc[phoneset_alpha_mapping.IPA==phone_ipa, 'liaphon'].values[0])

	return liaphon_sample




	



def main():
	args=build_arg_parser().parse_args()
	file_list=args.filelist
	outdir_path=args.outdir_path


	allosaurus_language=args.lang
	model_name=args.model_name




	model_path = get_model_path(model_name)
	inventory = Inventory(model_path)

	lang = allosaurus_language
	assert lang.lower() in inventory.lang_ids or lang.lower() in inventory.glotto_ids, f'language {lang} is not supported. Please verify it is in the language list'
	mask = inventory.get_mask(lang.lower(), approximation=False)

	unit = mask.target_unit
	phone_list=list(unit.id_to_unit.values())[1:]
	samples=[]

	with open(file_list,'r') as fl: 
		samples=[ sample.strip() for sample in fl.readlines() if os.path.exists(sample.strip())]
	n_samples=len(samples)

	config={
	# 'model':model,
	'lang':lang,
	'outdir':outdir_path,
	'phone_list':phone_list,
	'phoneset_map':"/mnt/mydrive/home/aghilas/Workspace/Experiments/SynPaFlex-Code/VoiceConversion/cvc-via-ppg/src/common/french_phoneset_mapping.csv",
	'acmod_path':args.acmod_path
	}
	if not os.path.exists(outdir_path):
		os.makedirs(outdir_path)


	print(config,samples)

	with mp.Pool(mp.cpu_count()) as p:
			gr2ph_align_sequences=list(
				tqdm(
					p.imap(
						run,
						zip(
							samples,	
							[config]*n_samples,
						
						),
					),
					total=n_samples,
				)
			)	

	for wf_copy in glob.glob(outdir_path+'/*.wav'):
		os.remove(wf_copy)



if __name__ == '__main__':
	main()