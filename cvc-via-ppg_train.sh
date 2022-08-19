#!/bin/bash

PROJECT_ROOT_DIR=${PWD}
KALDI_PATH=/vrac/software/kaldi/egs/timit/s5/


export PATH=$PATH:${PROJECT_ROOT_DIR}/tools/protoc-3.20.1-linux-x86_64/./bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PROJECT_ROOT_DIR}/tools/protoc-3.20.1-linux-x86_64/include/


export PYTHONPATH=$PROJECT_ROOT_DIR/src:$PYTHONPATH


#  Setup CUDA path.

export PATH=$PATH:/usr/local/cuda-10.1/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
                         


cd ${KALDI_PATH}


. ./path.sh






cd $PROJECT_ROOT_DIR




protoc -I=src/common --python_out=src/common src/common/data_utterance.proto





hparams=${1} #   hparams 




if test "$#" -ne 1; then 
	echo "##########################"
	echo "Usage:"
	echo "./cvc-via-ppg_train.sh <hparams> "
	exit 1
fi








echo "training ppg to mel model"
CUDA_VISIBLE_DEVICES=0 python -u ./src/script/train_ppg2mel.py ${hparams}
