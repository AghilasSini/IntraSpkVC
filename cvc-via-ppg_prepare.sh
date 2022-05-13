#!/bin/bash

PROJECT_ROOT_DIR=/mnt/mydrive/home/aghilas/Workspace/Experiments/SynPaFlex-Code/VoiceConversion/cvc-via-ppg
KALDI_PATH=/mnt/mydrive/home/aghilas/Workspace/tools/kaldi/egs/an4/s5


ENV_CONDA=/mnt/mydrive/home/aghilas/anaconda3/envs/pgg_extract
PRJ_ROOT=/gpfswork/rech/eyy/uyk62ct/asini/vcc20_baseline_cyclevae/baseline/lib
export PATH=$PATH:/mnt/mydrive/home/aghilas/Workspace/Experiments/SynPaFlex-Code/VoiceConversion/cvc-via-ppg/tools/protoc-3.19.4-linux-x86_64/./bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/mydrive/home/aghilas/Workspace/Experiments/SynPaFlex-Code/VoiceConversion/cvc-via-ppg/tools/protoc-3.19.4-linux-x86_64/include/


export PYTHONPATH=$PROJECT_ROOT_DIR/src:$PYTHONPATH


#  Setup Kaldi path.



#cd ${KALDI_PATH}


#. ./path






#cd $PROJECT_ROOT_DIR

#conda activate ${ENV_CONDA}



protoc -I=src/common --python_out=src/common src/common/data_utterance.proto





Wav_dir=${1} #   ${PROJECT_ROOT_DIR}/data_fr/wav/${mode}/${spk}
Out_dir=${2}  #{PROJECT_ROOT_DIR}/data_fr/filelists/${mode}/${spk}
spk=${3}


if test "$#" -ne 3; then 
	echo "##########################"
	echo "Usage:"
	echo "./cvc-via-ppg_prepare.sh <wav_dir> <out_dir> <spk_id> "
	exit 1
fi



if [ ! -d $Out_dir ];then
	mkdir -p ${Out_dir}
fi

# prepare data used for extracting ppg
echo "prepare data for ${spk}"
python -u ./src/common/gen_train_val_txt.py $Wav_dir $Out_dir $spk
echo "PPG extraction data preparation 1st step is done"
python -u ./src/script/extract_ppg.py ${spk}
