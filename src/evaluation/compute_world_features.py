import librosa
import pyworld
import numpy as np
import pysptk
from dtw import dtw, warp
from sklearn.metrics import mean_squared_error
import os

import argparse
import pandas as pd

# Values from Merlin: https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/vocoder/world/extract_features_for_merlin.sh
SAMPLING_RATE = 22050
MC_SIZE = 59
ALPHA = 0.65
import TextGrid





def get_silences_boundaries(textgrid_fpath,sampling_rate):
    
    try:
        if not os.path.exists(textgrid_fpath):
            raise FileNotExistException(textgrid_fpath)
        tg=TextGrid.TextGrid()
        tg.read(textgrid_fpath)
        for interval in tg['phonemes']:
            text=interval.mark()
            if not text or text== 'SIL':
                yield (int(interval.xmin()*sampling_rate),int(interval.xmax()*sampling_rate))
       
    except FileNotExistException as e:
        print(e)


   


def triming(filepath):
    texgrid_fpth=os.path.splitext(filepath)[0]+'.textgrid'
    wav, sampling_rate = librosa.load(filepath, sr=SAMPLING_RATE)

    wav_=wav.tolist()
    for ibeg,iend in get_silences_boundaries(texgrid_fpth,sampling_rate):
        if iend<len(wav_):
            del wav_[ibeg:iend]
        else:
            del wav_[ibeg:]
    
    return np.asarray(wav_)




def get_mgc(filepath):
    


    # triming silence at this stage
    wav=triming(filepath)

    # Perform WORLD Analysis
    f0, sp, ap = pyworld.wav2world(wav.astype(np.double), fs=SAMPLING_RATE)
    
    # Extracts the MGC coefficients
    mgc = pysptk.sptk.mcep(sp, order=MC_SIZE, alpha=ALPHA, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)


    return mgc, f0


# Compute the MCD as defined in https://www.cs.cmu.edu/~awb/papers/sltu2008/kominek_black.sltu_2008.pdf
# Note : In this implementation, the silences are not removed !
def compute_MCD(ref_mgc, synth_mgc):
    # intialize MCD
    mcd=0.0
    # If the samples are not of the same size, cut the longest
    if len(ref_mgc) > len(synth_mgc):
        ref_mgc = ref_mgc[:len(synth_mgc)]
    elif len(synth_mgc) > len(ref_mgc):
        synth_mgc = synth_mgc[:len(ref_mgc)]
    # print(ref_mgc.shape, synth_mgc.shape)
    try:
        # Get the indices where F0 is not 0 for at least one of the sequence    
        if len(ref_mgc)<1:
            raise ValueError('reference mgc sequence size is equal to 0')
        if len(synth_mgc)<1:
            raise ValueError('synthetic mgc sequence size is equal to 0')
    
        # Compute the MCD
        diff = ref_mgc - synth_mgc
        squared_diff = np.power(diff, 2)
        inner_sum = np.sum(squared_diff[:, 1:], axis=-1) # Removes the first dimension (ie. energy)
        root_inner_sum = np.sqrt(inner_sum)
        outer_sum = np.sum(root_inner_sum)
        mcd = 6.14185 / len(root_inner_sum) * outer_sum # The constant is defined in the paper
    except ValueError as ve:
        print(ve)



    return mcd


def compute_RMSE_F0(ref_seq, synth_seq):
    # initialize the rmse
    rmse=0.0

    # If the samples are not of the same size, cut the longest
    if len(ref_seq) > len(synth_seq):
        ref_seq = ref_seq[:len(synth_seq)]
    elif len(synth_seq) > len(ref_seq):
        synth_seq = synth_seq[:len(ref_seq)]
    ref_seq_no_zero = set((ref_seq != 0).nonzero()[0])
    synth_seq_no_zero = set((synth_seq != 0).nonzero()[0])
    indices_no_zero = list(ref_seq_no_zero.union(synth_seq_no_zero))
    try:
        # Get the indices where F0 is not 0 for at least one of the sequence    
        if len(ref_seq[indices_no_zero])<1:
            raise ValueError('reference f0 sequence size is equal to 0')
        if len(synth_seq[indices_no_zero])<1:
            raise ValueError('synthetic f0 sequence size is equal to 0')
        # compute the  rmse
        rmse = np.sqrt(mean_squared_error(ref_seq[indices_no_zero], synth_seq[indices_no_zero]))
        
    except ValueError as ve:
        print(ve)

    return rmse


# Raise this exception if a file is not valid
class FileNotExistException(Exception):
    def __init__(self,file_path,message="This file does not exists"):
        self.file_path = file_path
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return f'{self.file_path} -> {self.message}'










def objctive_evaluation(ref_files,synth_files,output_fpath):
    
    obj_measures={
        "ref_file":[], 
        'synth_file':[], 
        "mcd_wo_alignment":[], 
        "rmse_f0_wo_alignment":[], 
        "mcd_w_alignment":[], 
        "rmse_f0_w_alignment":[]
        }


    for ref_file, synth_file in  zip(ref_files,synth_files):
        #initialize dict 
        obj_measures['ref_file'].append(os.path.basename(ref_file))
        ref_mgc, ref_f0 = get_mgc(ref_file)

        obj_measures['synth_file'].append(os.path.basename(synth_file))
        synth_mgc, synth_f0 = get_mgc(synth_file)
        
        mcd_wo_alignment = compute_MCD(ref_mgc, synth_mgc)
        obj_measures['mcd_wo_alignment'].append(mcd_wo_alignment)

        rmse_f0_wo_alignment = compute_RMSE_F0(ref_f0, synth_f0)
        obj_measures['rmse_f0_wo_alignment'].append(rmse_f0_wo_alignment)

        alignment = dtw(ref_mgc, synth_mgc)
        if len(ref_mgc) > len(synth_mgc):
            aligned_ref_mgc = ref_mgc
            aligned_ref_f0 = ref_f0

            alignment_indexes = warp(alignment, index_reference=True)
            aligned_synth_mgc = synth_mgc[alignment_indexes]
            aligned_synth_f0 = synth_f0[alignment_indexes]
        else:
            aligned_synth_mgc = synth_mgc
            aligned_synth_f0 = synth_f0

            alignment_indexes = warp(alignment, index_reference=False)
            aligned_ref_mgc = ref_mgc[alignment_indexes]
            aligned_ref_f0 = ref_f0[alignment_indexes]

        mcd_w_alignment = compute_MCD(aligned_ref_mgc, aligned_synth_mgc)
        obj_measures['mcd_w_alignment'].append(mcd_w_alignment)

        rmse_f0_w_alignment = compute_RMSE_F0(aligned_ref_f0, aligned_synth_f0)
        obj_measures['rmse_f0_w_alignment'].append(rmse_f0_w_alignment)
    df=pd.DataFrame.from_dict(obj_measures)
    
    # Average mcd with alignment
    mcd_w_align=df['mcd_w_alignment'].to_list()
    mcd_w_avg_align=np.mean(mcd_w_align)
    mcd_w_std_align=np.std(mcd_w_align)

    # Average f0 with alignment
    rmse_f0_w_align=df['rmse_f0_w_alignment'].to_list()
    rmse_f0_w_avg_align=np.mean(rmse_f0_w_align)
    rmse_f0_w_std_align=np.std(rmse_f0_w_align)
    ##################################
    # Average mcd without alignment
    mcd_wo_align=df['mcd_wo_alignment'].to_list()
    mcd_wo_avg_align=np.mean(mcd_wo_align)
    mcd_wo_std_align=np.std(mcd_wo_align)

    # Average f0 with alignment
    rmse_f0_wo_align=df['rmse_f0_wo_alignment'].to_list()
    rmse_f0_wo_avg_align=np.mean(rmse_f0_wo_align)
    rmse_f0_wo_std_align=np.std(rmse_f0_wo_align)





    print('With Alignment : Avrage MCD  {:.3f} (+/-{:.3f}) (db)   Avrage RMSE F0 {:.3f} (+/-{:.3f}) (HZ)'.format(mcd_w_avg_align,mcd_w_std_align,rmse_f0_w_avg_align,rmse_f0_w_std_align))
    print('Without Alignemnt : Avrage MCD  {:.3f} (+/-{:.3f}) (db)   Avrage RMSE F0 {:.3f}(+/-{:.3f}) (HZ)'.format(mcd_wo_avg_align,mcd_wo_std_align,rmse_f0_wo_avg_align,rmse_f0_wo_std_align))
    df.to_csv(output_fpath,index=False)



def parser_build():
    parser = argparse.ArgumentParser(description="Sentence Spliter For French")
    parser.add_argument('ref_files', type=str, help='all data file')
    parser.add_argument('synth_waveglow', type=str, help='all data file')
    parser.add_argument('synth_griffinLim', type=str, help='all data file')
    parser.add_argument('automatic_measure_waveglow', type=str, help='all data file')
    parser.add_argument('automatic_measure_griffinLim', type=str, help='all data file')
    # parser.add_argument('--hparams',dest="hparams_file", default="default_hparams.json", type=str, help='all data file')
    return parser.parse_args()






if __name__ == "__main__":

    args=parser_build()
    ref_input_filepath = args.ref_files
    synth_waveglow_input_filepath =args.synth_waveglow 
    synth_griffinLim_input_filepath =args.synth_griffinLim #"/vrac/aperquin/test_tts/tal/misc/synth_griffinLim_files.txt"
    output_waveglow_filepath = args.automatic_measure_waveglow #"/vrac/aperquin/test_tts/tal/misc/automatic_measure_waveglow.tsv"
    output_griffinLim_filepath =args.automatic_measure_griffinLim #"/vrac/aperquin/test_tts/tal/misc/automatic_measure_griffinLim.tsv
    n_samples=-1
    # List the .wav files 
    with open(ref_input_filepath, 'r') as ref_input_file:
        ref_files = [line.strip() for line in ref_input_file.readlines()[:n_samples]]
    with open(synth_waveglow_input_filepath, 'r') as synth_waveglow_input_file:
        synth_waveglow_files = [line.strip() for line in synth_waveglow_input_file.readlines()[:n_samples]]
    with open(synth_griffinLim_input_filepath, 'r') as synth_griffinLim_input_file:
        synth_griffinLim_files = [line.strip() for line in synth_griffinLim_input_file.readlines()[:n_samples]]
    print(len(ref_files),len(synth_waveglow_files),len(synth_griffinLim_files))

    assert len(ref_files) == len(synth_waveglow_files) == len(synth_griffinLim_files)
    print()
    print('natural vs wavglow')
    mesures_waveglow=objctive_evaluation(ref_files,synth_waveglow_files,output_waveglow_filepath)
    print('natural vs griffinlim')
    mesures_griffinlim=objctive_evaluation(ref_files,synth_griffinLim_files,output_griffinLim_filepath)
    
            
            
