#!/bin/bash
#SBATCH --job-name=cvc_ppg2mel_train          # nom du job
#SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --ntasks=1                   # nombre total de taches (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
#SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference Ã  l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=./logout/cvc_ppg2mel_train%j.out      # nom du fichier de sortie
#SBATCH --error=./logout/cvc_ppg2mel_train%j.out       # nom du fichier d'erreur (ici commun avec la sortie)

# nettoyage des modules charges en interactif et herites par defaut
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

#kaldi path
cd /gpfswork/rech/eyy/commun/kaldi/egs/sre16/v2/
. ./path.sh
# anaconda  setup
source ~/.bashrc
# activate the repository
conda activate /gpfsscratch/rech/eyy/uyk62ct/anaconda/envs/ppg_speech

PROJECT_ROOT_DIR=/gpfswork/rech/eyy/uyk62ct/asini/cvc-via-ppg/
PRJ_ROOT=/gpfswork/rech/eyy/uyk62ct/asini/vcc20_baseline_cyclevae/baseline/lib
export PATH=$PATH:$PRJ_ROOT/flac/usr/local/bin/:$PRJ_ROOT/libsnd/usr/local/bin/:$SCRATCH/protoc/./bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PRJ_ROOT/flac/usr/local/lib/:$PRJ_ROOT/flac/usr/local/include/:$PRJ_ROOT/libsnd/usr/local/lib/:$PRJ_ROOT/libsnd/usr/local/include


export PYTHONPATH=$PROJECT_ROOT_DIR/src:$PYTHONPATH

cd $PROJECT_ROOT_DIR
spk="SiwisFrench"
hparam=$PROJECT_ROOT_DIR/data_fr/filelists/${spk}/hparams.json

protoc -I=src/common --python_out=src/common src/common/data_utterance.proto
echo "launch the train for the ${spk}"
python -u ./src/script/train_ppg2mel.py ${hparams}
