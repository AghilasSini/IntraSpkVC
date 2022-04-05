#!/bin/bash

PROJECT_ROOT_DIR=/mnt/mydrive/home/aghilas/Workspace/Experiments/SynPaFlex-Code/VoiceConversion/cvc-via-ppg
PRJ_ROOT=/gpfswork/rech/eyy/uyk62ct/asini/vcc20_baseline_cyclevae/baseline/lib
export PATH=$PATH:/mnt/mydrive/home/aghilas/Workspace/Experiments/SynPaFlex-Code/VoiceConversion/cvc-via-ppg/tools/protoc-3.19.4-linux-x86_64/./bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/mydrive/home/aghilas/Workspace/Experiments/SynPaFlex-Code/VoiceConversion/cvc-via-ppg/tools/protoc-3.19.4-linux-x86_64/include/


export PYTHONPATH=$PROJECT_ROOT_DIR/src:$PYTHONPATH

cd $PROJECT_ROOT_DIR


protoc -I=src/common --python_out=src/common src/common/data_utterance.proto
echo "this is how we do?"
# mode='Pronunciation'
# spk="neb_syn"

Wav_dir=${1} #   ${PROJECT_ROOT_DIR}/data_fr/wav/${mode}/${spk}
Out_dir=${2}  #{PROJECT_ROOT_DIR}/data_fr/filelists/${mode}/${spk}
spk=${3}
# if Out_dir

if [ -d $Out_dir ];then
	mkdir -p ${Out_dir}
fi

# prepare data used for extracting ppg
echo "prepare data for ${spk}"
python -u ./src/common/gen_train_val_txt.py $Wav_dir $Out_dir $spk
# echo "PPG extraction data preparation 1st step is done"
# python -u ./src/script/extract_ppg.py ${mode}/${spk} 
