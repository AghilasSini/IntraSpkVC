

# System/default
import sys
import os

# Arguments
import argparse

# Messaging/logging
import traceback
import time
import logging

# Data/plot
import numpy as np
import pandas as pd
# import plotnine as p9

# Warning
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("error", RuntimeWarning)

###############################################################################
# global constants
###############################################################################
LEVEL = [logging.WARNING, logging.INFO, logging.DEBUG]
BG_COLOR = '#FFFFFF'


###############################################################################
# Functions
###############################################################################




class FileNotExistException(Exception):
	def __init__(self,file_path,message="This file does not exists"):
		self.file_path = file_path
		self.message = message
		super().__init__(self.message)
	def __str__(self):
		return f'{self.file_path} -> {self.message}'

###############################################################################
# Main function
###############################################################################
def main():
	"""Main entry function
	"""
	global args





###############################################################################
#  Envelopping
###############################################################################
if __name__ == '__main__':
	try:
		parser = argparse.ArgumentParser(description="")

		# Add options
		parser.add_argument("-l", "--log_file", default=None,
							help="Logger file")
		parser.add_argument("-v", "--verbosity", action="count", default=0,
							help="increase output verbosity")
			# Add arguments
		parser.add_argument("filelist", help="The input file to be projected")
		parser.add_argument("inputdir", help="The input file to be projected")


		# Parsing arguments
		args = parser.parse_args()

		# create logger and formatter
		logger = logging.getLogger()
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

		# Verbose level => logging level
		log_level = args.verbosity
		if (args.verbosity >= len(LEVEL)):
			log_level = len(LEVEL) - 1
			logger.setLevel(log_level)
			logging.warning("verbosity level is too high, I'm gonna assume you're taking the highest (%d)" % log_level)
		else:
			logger.setLevel(LEVEL[log_level])

		# create console handler
		ch = logging.StreamHandler()
		ch.setFormatter(formatter)
		logger.addHandler(ch)

		# create file handler
		if args.log_file is not None:
			fh = logging.FileHandler(args.log_file)
			logger.addHandler(fh)

		# Debug time
		start_time = time.time()
		logger.info("start time = " + time.asctime())

		# Running main function <=> run application
		main()

		# Debug time
		logging.info("end time = " + time.asctime())
		logging.info('TOTAL TIME IN MINUTES: %02.2f' %
					 ((time.time() - start_time) / 60.0))

		# Exit program
		sys.exit(0)
	except KeyboardInterrupt as e:  # Ctrl-C
		raise e
	except SystemExit:  # sys.exit()
		pass
	except Exception as e:
		logging.error('ERROR, UNEXPECTED EXCEPTION')
		logging.error(str(e))
		traceback.print_exc(file=sys.stderr)
		sys.exit(-1)