import torch
from glob import glob
import pickle
def main():
	invalid_files=[]
	for fl in glob('/vrac/asini/workspace/voice_conversion/IntraSpkVC/data_fr/mel/Nadine/*.pkl'):
		with open(fl,'rb') as openFile:
			data=pickle.load(openFile)
			print(data.shape)
			if data.shape[0]<1 or data.shape[1]!=80:
				invalid_files.append(fl)

	print(invalid_files)
if __name__ == '__main__':
	main()


