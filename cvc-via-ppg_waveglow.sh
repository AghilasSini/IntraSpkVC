#!/bin/bash

PROJECT_ROOT_DIR=${PWD}
KALDI_PATH=/vrac/software/kaldi/egs/timit/s5/


export PATH=$PATH:${PROJECT_ROOT_DIR}/tools/protoc-3.20.1-linux-x86_64/./bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PROJECT_ROOT_DIR}/tools/protoc-3.20.1-linux-x86_64/include/


export PYTHONPATH=$PROJECT_ROOT_DIR/src:$PYTHONPATH


#  Setup CUDA path.

#export PATH=$PATH:/usr/local/cuda-10.1/bin

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
                         


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

test_data_set=/vrac/asini/workspace/voice_conversion/IntraSpkVC/data_fr/filelists/${3}/test_set.txt
if [ -f $teacher_utterance_path ];then
	rm $teacher_utterance_path
fi

for fl in `cat ${test_data_set}`;do
	testFile=/vrac/asini/workspace/voice_conversion/IntraSpkVC/data_fr/ppg_reduce/`echo ${3}|cut -d _ -f 1`/`basename ${fl%.wav}`_ppg.npy
	#echo $testFile
	if [ -f $testFile ];then
		echo $testFile >> $teacher_utterance_path
	fi

done



if [ ! -f ${output_dir} ];then
	mkdir -p ${output_dir}
fi

waveglow_model=/vrac/lwadoux/vocodeurs/waveglow_multi_256_tal

#waveglow_model=/vrac/lwadoux/vocodeurs/wg_nadine
#CUDA_VISIBLE_DEVICES=0 python -u ./src/script/waveglow_synth.py  --ppg ${teacher_utterance_path}  --checkpoint ${ppg2mel_model} --waveglow ${waveglow_model}  --hparams ${hparams}   --out_dirname ${output_dir}
python -u ./src/script/waveglow_synth.py  --ppg ${teacher_utterance_path}  --checkpoint ${ppg2mel_model} --waveglow ${waveglow_model}  --hparams ${hparams}   --out_dirname ${output_dir}

#CUDA_VISIBLE_DEVICES=0 python -u ./src/script/train_ppg2mel.py ${hparams}
