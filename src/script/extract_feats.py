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

import plotnine as p9

# Argument
import argparse

# Debug
import traceback
import time

import codecs

from tqdm import tqdm
import multiprocessing as mp
from scipy.io import wavfile
import time
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
	parser.add_argument("ivector_dir", help="The input file to be projected")
	parser.add_argument("mfcc_dir", help="The input file to be projected")
	parser.add_argument("ppg_ivSpk_mfccSpk", help="outdir to be projected")
	
	args = parser.parse_args()
	return args



			
def main():
	
	args=args_parser()

	ivector_dir=args.ivector_dir
	mfcc_dir=args.mfcc_dir

	for mfcc, ivect in zip(ivector_dir,mfcc_dir):
		np.read




if __name__ == '__main__':
	main()




# PPG extraction 
# args @ivector_dir
#	   @mfcc_dir
# output
#      @ppg_ivSpk_mfccSpk
