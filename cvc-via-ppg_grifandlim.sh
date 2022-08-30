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









if test "$#" -ne 3; then 
	echo "##########################"
	echo "Usage:"
	echo "./cvc-via-ppg_grifandlim.sh <ppg_mode_path> <sample_ppg.npy> <target_speaker> "
	exit 1
fi




ppg2mel_model=${1}

teacher_utterance_path=${2}
output_dir=$PROJECT_ROOT_DIR/data_fr/synth/${3}

hparams=/vrac/asini/workspace/voice_conversion/IntraSpkVC/data_fr/filelists/${3}/hparams.json

echo "generate sample using ${ppg2mel_model}"

CUDA_VISIBLE_DEVICES=0 python -u ./src/script/griffandlim_synth.py  --ppg ${teacher_utterance_path}  --checkpoint ${ppg2mel_model}   --hparams ${hparams}   --out_dirname ${output_dir}  

#CUDA_VISIBLE_DEVICES=0 python -u ./src/script/train_ppg2mel.py ${hparams}
