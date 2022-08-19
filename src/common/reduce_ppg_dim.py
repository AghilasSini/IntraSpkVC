#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Baseline
import sys
import os

# Argument
import argparse

# Debug
import traceback
import time

# Logging
import logging
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]

# Numerical / db
import numpy as np
import pandas as pd

# PCA part
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Plotting
# import plotnine as p9
# import matplotlib.pyplot as plt


# Ignore some warning
import warnings
import matplotlib
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)


# kaldi libs
from kaldi.matrix import Matrix, Vector
from kaldi.alignment import  Aligner
import numpy as np



# import seaborn as sns

# import pandas as pd 
# import re
# import sys
# import os
# import codecs
# import argparse


# import multiprocessing as mp
# from tqdm import tqdm






def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def softmax(X, theta = 1.0, axis = None):
	"""
	Compute the softmax of each element along an axis of X.

	Parameters
	----------
	X: ND-Array. Probably should be floats.
	theta (optional): float parameter, used as a multiplier
		prior to exponentiation. Default = 1.0
	axis (optional): axis to compute values along. Default is the
		first non-singleton axis.

	Returns an array the same size as X. The result will sum to 1
	along the specified axis.
	"""

	# make X at least 2d
	y = np.atleast_2d(X)

	# find axis
	if axis is None:
		axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

	# multiply y against the theta parameter,
	y = y * float(theta)

	# subtract the max for numerical stability
	y = y - np.expand_dims(np.max(y, axis = axis), axis)

	# exponentiate y
	y = np.exp(y)

	# take the sum along the specified axis
	ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

	# finally: divide elementwise
	p = y / ax_sum

	# flatten if X was 1D
	if len(X.shape) == 1: p = p.flatten()

	return p


def get_pdf_to_phones_map(trans_model):

	ret = [[] for _ in range(trans_model.num_pdfs())]
	print("numero of pdf {} ".format(len(ret)))
	n_trans_id = trans_model.num_transition_ids()
	trans_id = 0
	pdf_id = 0
	# ret=[]
	for i in range(n_trans_id):
		trans_id = i + 1  # trans-id is one-based, https://kaldi-asr.org/doc/hmm.html#transition_model_identifiers
		pdf_id = trans_model.transition_id_to_pdf(trans_id)
		phone = trans_model.transition_id_to_phone(trans_id)
		# hmm_state = trans_model.transition_id_to_hmm_state(trans_id)
		ret[pdf_id].append(phone)
	return ret


def  plpp(n_phones, prob_row, pdf2phones):
	ret = np.zeros(n_phones)
	for i in range(len(prob_row)):
		dest_idxs = set(np.asarray(pdf2phones[i]) - 1)
		# print(len(dest_idxs))
		for idx in dest_idxs:
			ret[idx] += prob_row[i]
	return np.log(ret)


def build_arg_parser():
	parser = argparse.ArgumentParser(description="")
	# Add options
	parser.add_argument("-v", "--verbosity", action="count", default=0,help="increase output verbosity")


	parser.add_argument("pgg_filepath", help="audio file")
	parser.add_argument("output_dir_fpath", help="output directory")
	parser.add_argument('--language',dest='lang',default='fr')
	parser.add_argument('--topk',dest='top_k',type=int,default=5)
	parser.add_argument('--model_path',dest='acmod',default='/vrac/asini/workspace/voice_conversion/IntraSpkVC/data_fr/utils/final.mdl', help='models directory')
	parser.add_argument('--phone_set',dest='phoneset_path',default='/vrac/asini/workspace/voice_conversion/IntraSpkVC/data_fr/utils/true_phone.txt', help='models directory')
	return parser



def main():
	try:
		args=build_arg_parser().parse_args()
		ppg_full_fpath=args.pgg_filepath
		ppg_redu_fpath=os.path.join(args.output_dir_fpath,os.path.basename(ppg_full_fpath))

		 # Verbose level => logging level
		log_level = args.verbosity
		print(log_level)
		if (args.verbosity >= len(LEVEL)):
			log_level = len(LEVEL) - 1
			logging.basicConfig(level=LEVEL[log_level])
			logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)
		else:
			print('logging data')
			logging.basicConfig(level=LEVEL[log_level])

		start_time = time.time()
		logging.info("start time = " + time.asctime())
		# print("start time :{:.3f}".format(time.asctime()))

		sample_ppg=np.load(ppg_full_fpath)
		trans_model_path=args.acmod
		n_phones=259
		trans_model=Aligner.read_model(trans_model_path)
		print("after loading the model  {:.3f}".format((time.time() - start_time)/60))



		pdf2phones=get_pdf_to_phones_map(trans_model)
		print("after loading the model  {:.3f}".format((time.time() - start_time)/60))
		# arr = np.array(pdf2phones)
		# print(arr.shape)
		an_array = np.zeros((sample_ppg.shape[0], n_phones))
		for id_row,prob_row in enumerate(sample_ppg):
			ppg_prob_reduce=plpp(n_phones,prob_row,pdf2phones)
			
			ppg_prob_reduce_softmax=softmax(ppg_prob_reduce)
			an_array[id_row]= ppg_prob_reduce_softmax
			# print(np.sum(ppg_prob_reduce))
		num_frames=an_array.shape[0]


	
		with open(ppg_redu_fpath,'wb') as ppg_red:
			np.save(ppg_red,an_array)
			# Debug time
		print(" ending processing {:.3f}".format((time.time() - start_time) / 60.0))
		logging.info("end time = " + time.asctime())
		logging.info('TOTAL TIME IN MINUTES: %02.2f' %
					 ((time.time() - start_time) / 60.0))

		# Exit program
		sys.exit(0)
	except KeyboardInterrupt as e:  # Ctrl-C
		raise e
	except SystemExit as e:  # sys.exit()
		pass
	except Exception as e:
		logging.error('ERROR, UNEXPECTED EXCEPTION')
		logging.error(str(e))
		traceback.print_exc(file=sys.stderr)
		sys.exit(-1)
	#new_array= an_array.T   ##np.zeros((n_phones,num_frames))
	#print(new_array.shape)




if __name__ == '__main__':
	main()
