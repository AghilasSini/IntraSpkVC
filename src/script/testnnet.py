import kaldi
import os
from common import decode
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data_fr')
NNET_PATH = os.path.join(DATA_DIR, 'am', 'final.raw')


nnet = decode.read_nnet3_model(NNET_PATH)
print(kaldi.nnet3.nnet_info(nnet))
