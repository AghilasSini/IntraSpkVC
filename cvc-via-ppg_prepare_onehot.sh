#!/bin/bash

PROJECT_ROOT_DIR=${PWD}
KALDI_PATH=/vrac/software/kaldi/egs/timit/s5/


export PATH=$PATH:${PROJECT_ROOT_DIR}/tools/protoc-3.20.1-linux-x86_64/./bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PROJECT_ROOT_DIR}/tools/protoc-3.20.1-linux-x86_64/include/


export PYTHONPATH=$PROJECT_ROOT_DIR/src:$PYTHONPATH


#  Setup Kaldi path.



cd ${KALDI_PATH}


. ./path.sh





# Features extraction

cd $PROJECT_ROOT_DIR

export PYTHONPATH=$PROJECT_ROOT_DIR/src:$PYTHONPATH


protoc -I=src/common --python_out=src/common src/common/data_utterance.proto





spk=${1}




if test "$#" -ne 1; then 
	echo "##########################"
	echo "Usage:"
	echo "./cvc-via-ppg_prepare.sh   <spk_id>"
	exit 1
fi






file_wav_list=${PROJECT_ROOT_DIR}/data_fr/filelists/${spk}/file_id_wav.scp

out_dir=${PROJECT_ROOT_DIR}/data_fr/one_hot/${spk}
if [ ! -d  ${out_dir} ];then
	mkdir -p ${out_dir}
fi

python ./src/script/embedding.py ${file_wav_list}   ${out_dir}
