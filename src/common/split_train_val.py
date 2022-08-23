import sys
import os
from random import choice
import librosa

import sys
import os
import pandas 
import argparse

from sklearn.model_selection import train_test_split

import json
import codecs
import numpy

class DataPrepare(object):
	"""docstring for DataPrepare"""
	def __init__(self,file_id_list,output_dir_fpath,speaker_id,default_hparams):
		super(DataPrepare, self).__init__()
		self.file_id_list = file_id_list
		self.speaker_id=speaker_id
		self.output_dir_fpath=os.path.join(output_dir_fpath,speaker_id)
		self.default_hparams=default_hparams
		with open(default_hparams) as infile:
			self.hparams_dict=json.load(infile)

	def run(self,model_output_dir,n_symbols=3536,is_finetuning=False, checkpoint_path=None):
		list_valid_list=self.get_file_list()
		if not os.path.exists(self.output_dir_fpath):
			os.makedirs(self.output_dir_fpath)
		if len(list_valid_list)>1:
			train_size=self.generate_training_validation_txt (list_valid_list)
			self.generate_spk2utt(list_valid_list)
			self.generate_wav_scp(list_valid_list)
			self.generate_wav_list(list_valid_list)
			self.update_default_hparams(model_output_dir,train_size)
		else:
			print("something wrong")

	def get_file_list (self):
		df=pandas.read_csv(self.file_id_list,sep='|')
		
		return [ fl for fl in  df['clips_with_full_path'].to_list() if os.path.exists(fl)]

	# def valid_audio_file(fl):
		

	def generate_training_validation_txt (self,list_valid_list):
	
		train,test, _,_ = train_test_split(list_valid_list, list_valid_list, test_size=0.05, random_state=42)
	
		training_files_list=os.path.join(self.output_dir_fpath, "training_set.txt")


		with open(training_files_list, "w") as train_file:
			for elt in train:
				train_file.write(elt+ "\n") 
		
		# update training  set 
		self.hparams_dict['training_files']=training_files_list


		validation_files_list=os.path.join(self.output_dir_fpath, "validation_set.txt")
		with open(validation_files_list, "w") as test_file:
			for elt in test:
				test_file.write(elt+"\n")
		# update validation set
		self.hparams_dict['validation_files']=validation_files_list

		return len(train)
		


	def generate_spk2utt (self,list_valid_list):
		dest=os.path.join(self.output_dir_fpath,"spk2utt")
		with open(dest, "w") as s2u_file:
			s2u_file.write(self.speaker_id + " ")
			for elt in list_valid_list:
				name = os.path.basename(elt.split(".wav")[0])
				s2u_file.write(name + " ")
		
	def generate_wav_scp (self,list_valid_list):
		dest=os.path.join(self.output_dir_fpath,"wav.scp")
		with open(dest, "w") as wav_file:
			for elt in list_valid_list:
				name = os.path.basename(elt.split(".wav")[0])
				wav_file.write(name + " " + elt + "\n")

	def generate_wav_list(self,list_valid_list):
		dest=os.path.join(self.output_dir_fpath,"file_id_wav.scp")
		with open(dest, "w") as wav_file:
			for elt in list_valid_list:
				wav_file.write("{}\n".format(elt))

	def update_default_hparams(self,model_output_dir,train_set_size,n_symbols=3536,is_finetuning=False, checkpoint_path=None):
		
		if is_finetuning and os.path.exists(checkpoint_path):
			self.hparams_dict["checkpoint_path"]=checkpoint_path
			self.hparams_dict["use_saved_learning_rate"]=True
		if n_symbols!=self.hparams_dict["n_symbols"]:
			self.hparams_dict["n_symbols"]=n_symbols
		
		# self.hparams_dict["batch_size"]=12
		# self.hparams_dict["hop_length"]=221
		# self.hparams_dict["sampling_rate"]=22050

		self.hparams_dict['iters_per_checkpoint']=numpy.ceil(  train_set_size  /self.hparams_dict['batch_size'])

		self.hparams_dict["output_directory"]=os.path.join(model_output_dir,"models/"+self.speaker_id)
		self.hparams_dict["log_directory"]=os.path.join(model_output_dir,"models/"+self.speaker_id+"/log")
		self.hparams_dict['ppg_fdir']=os.path.join(model_output_dir,"ppg/"+self.speaker_id)
		self.hparams_dict['mel_fdir']=os.path.join(model_output_dir,"mel/"+self.speaker_id)
		output_hparams_file=os.path.join(self.output_dir_fpath,"hparams.json")

		with codecs.open(output_hparams_file, 'w','utf-8') as outfile:
			json.dump(self.hparams_dict,outfile)







def parser_build():
	parser = argparse.ArgumentParser(description="Sentence Spliter For French")
	parser.add_argument('WAV_FILES_FOLDER', type=str, help='all data file')
	parser.add_argument('OUTPUT_FOLDER', type=str, help='all data file')
	parser.add_argument('MODEL_SAVE_DIR', type=str, help='all data file')
	parser.add_argument('SPK_NAME', type=str, help='all data file')
	parser.add_argument('--hparams',dest="hparams_file", default="default_hparams.json", type=str, help='all data file')
	return parser.parse_args()

def main():

	args=parser_build()
	file_id_list=args.WAV_FILES_FOLDER
	out_dir_fpath=args.OUTPUT_FOLDER
	model_save_dir=args.MODEL_SAVE_DIR
	speaker_id=args.SPK_NAME
	hparams_file=args.hparams_file
	data_prepare=DataPrepare(file_id_list,out_dir_fpath,speaker_id,hparams_file)
	
	data_prepare.run(model_save_dir)

	


if __name__ == "__main__":
	main()
	
