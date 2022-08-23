#!/bin/bash
#SBATCH --job-name=vc-ppg_prepare
#SBATCH --partition=prepost
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:30:00
#SBATCH --output=./logout/vc-ppg_prepare_prepare.out
#SBATCH --error=./logout/vc-ppg_prepare_prepare.log

# module loading
module purge
module load git/2.25.0
module load llvm/10.0.1
module load gcc/8.3.1
module load cuda/10.2
module load nccl/2.8.3-1-cuda
module load cudnn/8.0.4.30-cuda-10.2
module load intel-mkl/2020.4
module load magma/2.5.4-cuda
module load openmpi/4.0.5-cuda
#module load anaconda-py3/2019.03
#module load cuda/10.1.1

set -x


# anaconda  setup
source ~/.bashrc

anaconda_dir=/gpfsscratch/rech/eyy/uyk62ct/anaconda
# activate the repository
conda activate ${anaconda_dir}/envs/ppg_speech

PROJECT_ROOT_DIR=/gpfswork/rech/eyy/uyk62ct/asini/IntraSpkVC
PRJ_ROOT=/gpfswork/rech/eyy/uyk62ct/asini/vcc20_baseline_cyclevae/baseline/lib
export PATH=$PATH:$PRJ_ROOT/flac/usr/local/bin/:$PRJ_ROOT/libsnd/usr/local/bin/:$SCRATCH/protoc/./bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRJ_ROOT/flac/usr/local/lib/:$PRJ_ROOT/flac/usr/local/include/:$PRJ_ROOT/libsnd/usr/local/lib/:$PRJ_ROOT/libsnd/usr/local/include:$SCRATCH/protoc/include/





#  Setup Kaldi path.

KALDI_PATH=/gpfswork/rech/eyy/commun/kaldi/egs/sre16/v2/

cd ${KALDI_PATH}


. ./path.sh






cd $PROJECT_ROOT_DIR




protoc -I=src/common --python_out=src/common src/common/data_utterance.proto



#"data_Victoria_CorrV2.1.tsv"
#"data_ReneDepasse_CorrV2.1.tsv"
#"data_Nadine_CorrV2.tsv"
#"data_Jean-LucFischer_CorrV2.tsv"
csv_data_fpath="/vrac/mufasa/v0/data_SiwisFrench_CorrV2.tsv" #    ${PROJECT_ROOT_DIR}/data_fr/wav/${mode}/${spk}
Out_dir=${PROJECT_ROOT_DIR}/data_fr/filelists #/${mode}/${spk}
Model_dir=${PROJECT_ROOT_DIR}/data_fr/
spk="SiwisFrench"




#if test "$#" -ne 2; then 
#	echo "##########################"
#	echo "Usage:"
#	echo "./cvc-via-ppg_prepare.sh <csv_data_fpath>   <spk_id> "
#	exit 1
#fi



if [ ! -d $Out_dir ];then
	mkdir -p ${Out_dir}
fi


config_fpath=${PROJECT_ROOT_DIR}/src/waveglow/config.json


# prepare data used for extracting ppg
echo "prepare data for ${spk}"
python -u ./src/common/split_train_val.py $csv_data_fpath $Out_dir $Model_dir $spk --hparams ${PROJECT_ROOT_DIR}/data_fr/default_hparams.json
echo "PPG extraction data preparation 1st step is done"
python -u ./src/script/extract_ppg.py ${spk}
echo "Extract acoustic features MelSpectrogram"
ls  ${PROJECT_ROOT_DIR}/data_fr/wav/${spk}/*.wav >${PROJECT_ROOT_DIR}/data_fr/filelists/${spk}/file_id_wav.scp
file_wav_list=${PROJECT_ROOT_DIR}/data_fr/filelists/${spk}/file_id_wav.scp
python ./src/waveglow/mel2samp.py -f ${file_wav_list} -c ${config_fpath} -o ${PROJECT_ROOT_DIR}/data_fr/mel/${spk}
