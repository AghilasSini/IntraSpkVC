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









if test "$#" -ne 2; then 
	echo "##########################"
	echo "Usage:"
	echo "./cvc-via-ppg_analyse-synthesis.sh <file_list> <speaker_id>"
	exit 1
fi




#ppg2mel_model=${1}
waveglow_model=/vrac/software/waveglow/waveglow_256channels_universal_v5.pt
config_fpath=/vrac/asini/workspace/voice_conversion/IntraSpkVC/src/waveglow/config.json
file_wav_list=${1}
output_dir=$PROJECT_ROOT_DIR/data_fr/synth/${2}

hparams=/vrac/asini/workspace/voice_conversion/IntraSpkVC/data_fr/filelists/${2}/hparams.json

echo "generate sample using ${ppg2mel_model}"

# python ./src/waveglow/mel2samp.py -f ${file_wav_list} -c ${config_fpath} -o ${output_dir}
file_mel_list=${file_wav_list%.txt}_mel.txt
ls ${output_dir}/*.pt >$file_mel_list

# for fl in  `cat $file_wav_list`;do
# 	echo ${output_dir}/`basename ${fl}`.pt
# done
# cat ${file_mel_list}
CUDA_VISIBLE_DEVICES=0 python ./src/waveglow/inference.py -f ${file_mel_list} -w $waveglow_model -o $output_dir  
