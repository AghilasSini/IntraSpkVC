import os
import sys

# import torch
# from PIL import Image
# import numpy as np
# from glob import glob




sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from common import  decode
from common.utterance import Utterance,numpy_to_mat
#from common.data_utils import get_ppg,utt_to_sequence

from ppg import compute_full_ppg_chain,reduce_ppg_dim,compute_full_ppg_softmax

from kaldi.base.io import ofstream




from kaldi.util.table import SequentialMatrixReader
from kaldi.matrix.sparse import SparseMatrix

from scipy.io import wavfile
from common.data_utterance_pb2 import DataUtterance, Segment, MetaData,VocoderFeature,FloatMatrix
from numpy import save


from sklearn.preprocessing import normalize

# static
num_senones=3536
num_phonemes=40
nzero_prob=0.01

# data path
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data_fr/')
SPK_DIR=os.path.join(DATA_DIR,'filelists/{}'.format(sys.argv[1]))

NNET_PATH = os.path.join(DATA_DIR, 'am', 'final.raw')
MFCC_CONF_PATH=os.path.join(DATA_DIR, 'conf', 'mfcc_hires.conf')
WAV_SCP_PATH=os.path.join(SPK_DIR, 'wav.scp')
IVECTOR_CONF_PATH=os.path.join(DATA_DIR, 'conf', 'ivector_extractor.conf')
SPK2UTT_PATH=os.path.join(SPK_DIR, 'spk2utt')
# bash command line fro extracting MFCC features and ivectors

RESULT_DIR_PATH=os.path.join(DATA_DIR, 'ppg/{}'.format(sys.argv[1]))

if not os.path.exists(RESULT_DIR_PATH):
    os.makedirs(RESULT_DIR_PATH)

feats_rspec = ("ark:compute-mfcc-feats --config={} "
        "scp:{}  ark:- |".format(MFCC_CONF_PATH,WAV_SCP_PATH))

ivectors_rspec = (feats_rspec + "ivector-extract-online2 "
                          "--config={} "
                                            "ark:{} ark:- ark:- |".format(IVECTOR_CONF_PATH,SPK2UTT_PATH))


with SequentialMatrixReader(feats_rspec) as feats_reader, \
             SequentialMatrixReader(ivectors_rspec) as ivectors_reader:
                 for (fkey, feats), (ikey, ivectors) in zip(feats_reader, ivectors_reader):
                             assert(fkey == ikey)
                             # concatenate "feats" mfcc with ivectors ....
                             #feats = feats/np.linalg.norm(feats)
                             #ivectors = ivectors/np.linalg.norm(feats)
                             
                             #np.savetxt(os.path.join(RESULT_DIR_PATH, "{}_ivects.csv".format(fkey)), ivectors, delimiter=",")
                             #np.savetxt(os.path.join(RESULT_DIR_PATH, "{}_feats.csv".format(fkey)), feats, delimiter=",")

                             feat_tuple=(feats,ivectors)
                             # load acoustic model
                             nnet = decode.read_nnet3_model(NNET_PATH)


                             # extract ppg from chain model
                             ppgs=compute_full_ppg_chain(nnet,feat_tuple)
                             # create transform sparce matrix
                             transform=SparseMatrix().from_dims(num_phonemes,num_senones).set_randn_(nzero_prob)
     
                             # reduce from senones to  monophone
                             reduce_ppg=reduce_ppg_dim(ppgs, transform)
                             output_full_path=os.path.join(RESULT_DIR_PATH,"{}_ppg".format(fkey))
                             ppg_numpy_style=ppgs.numpy()
                             save(output_full_path,ppg_numpy_style)
 #                           mat_pggs=FloatMatrix()
#                             numpy_to_mat(ppg_numpy_style,mat_pggs)
                       



# second
#for teacher_utt_path in glob('/commun/bdd/nadine/FR/fr/aghilas/aghilas_*.wav'):
#    utt = Utterance()
#    fs, wav = wavfile.read(teacher_utt_path)
#    utt.fs = fs
#    utt.wav = wav
#    utt.ppg = get_ppg(teacher_utt_path, deps)
#    out_=utt_to_sequence(utt,is_full_ppg=True)
#    print("  data out  {}".format(out_.shape))
