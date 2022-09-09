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




ppg_dir=${1}
ppg_reduce_dir=${2}
spk_id=${3}




if test "$#" -ne 3; then 
	echo "##########################"
	echo "Usage:"
	echo "./cvc-via-ppg_prepare.sh <ppg_dir> <ppg_reduce_dir>   <spk_id> "
	exit 1
fi



if [ ! -d $Out_dir ];then
	mkdir -p ${Out_dir}
fi


config_fpath=${PROJECT_ROOT_DIR}/src/waveglow/config.json






if [ ! -d $ppg_reduce_dir/${spk_id} ]; then
	mkdir -p $ppg_reduce_dir/${spk_id}
fi

# prepare data used for extracting ppg
echo "prepare data for ${spk}"
python -u ./src/common/reduce_ppg_dim.py ${ppg_dir} ${ppg_reduce_dir}/${spk_id} --model_path $PROJECT_ROOT_DIR/data_fr/utils/final.mdl --phone_set ${PROJECT_ROOT_DIR}/data_fr/utils/true_phone.txt
echo "update hparams"

python -u ./src/common/hparams_updater.py --hparams ${PROJECT_ROOT_DIR}/data_fr/filelists/${spk_id}/hparams.json

