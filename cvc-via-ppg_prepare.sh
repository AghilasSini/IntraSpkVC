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





Wav_dir=${1} #   ${PROJECT_ROOT_DIR}/data_fr/wav/${mode}/${spk}
Out_dir=${2}  #{PROJECT_ROOT_DIR}/data_fr/filelists/${mode}/${spk}
Model_dir=${3}
spk=${4}




if test "$#" -ne 4; then 
	echo "##########################"
	echo "Usage:"
	echo "./cvc-via-ppg_prepare.sh <wav_dir> <out_dir> <model_dir>  <spk_id> "
	exit 1
fi



if [ ! -d $Out_dir ];then
	mkdir -p ${Out_dir}
fi


config_fpath=/vrac/asini/workspace/voice_conversion/IntraSpkVC/src/waveglow/config.json


# prepare data used for extracting ppg
echo "prepare data for ${spk}"
python -u ./src/common/split_train_val.py $Wav_dir $Out_dir $Model_dir $spk --hparams ${PROJECT_ROOT_DIR}/data_fr/default_hparams.json
echo "PPG extraction data preparation 1st step is done"
python -u ./src/script/extract_ppg.py ${spk}
echo "Extract acoustic features MelSpectrogram"
#ls  ${PROJECT_ROOT_DIR}/data_fr/wav/${spk}/*.wav >${PROJECT_ROOT_DIR}/data_fr/filelists/${spk}/file_id_wav.scp
file_wav_list=${PROJECT_ROOT_DIR}/data_fr/filelists/${spk}/file_id_wav.scp
python ./src/waveglow/mel2samp.py -f ${file_wav_list} -c ${config_fpath} -o ${PROJECT_ROOT_DIR}/data_fr/mel/${spk}
