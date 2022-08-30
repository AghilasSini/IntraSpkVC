# Copyright 2019 Guanlong Zhao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from common.data_utils import get_ppg
from common.hparams import create_hparams_stage,HParamsView
from common.layers import TacotronSTFT
from common.utils import waveglow_audio, get_inference, load_waveglow_model
from scipy.io import wavfile
from script.train_ppg2mel import load_model
from waveglow.denoiser import Denoiser
from common.audio_processing import griffin_lim
from common.stft import STFT
import argparse
import logging
import os
import ppg
import torch
import numpy as np
import json
import librosa

import soundfile



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate accent conversion speech using pre-trained'
                    'models.')
    parser.add_argument('--ppg2mel_model', type=str, required=True,
                        help='Path to the PPG-to-Mel model.')
    parser.add_argument('--waveglow_model', type=str, required=True,
                        help='Path to the WaveGlow model.')
    parser.add_argument('--teacher_utterance_path', type=str, required=True,
                        help='Path to a native speaker recording.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output dir, will save the audio and log info.')
    parser.add_argument('--extract_mel', type=bool, required=False,
                        help='Save or not converted mel-spectrogram to output folder.')
    parser.add_argument('--hparams', type=str, required=False,
                        help='hyper-parameters json file')
    args = parser.parse_args()

    # Prepare dirs
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    logging.basicConfig(filename=os.path.join(output_dir, 'debug.log'),
                        level=logging.DEBUG)
    logging.info('Output dir: %s', output_dir)

    # Parameters
    teacher_utt_path = args.teacher_utterance_path
    checkpoint_path = args.ppg2mel_model
    waveglow_path = args.waveglow_model
    is_clip = False  # Set to True to control the output length of AC.
    fs = 22050
    waveglow_sigma = 1.0
    waveglow_for_denoiser = torch.load(waveglow_path)['model']

    waveglow_for_denoiser.cuda()
    denoiser_mode = 'zeros'
    denoiser = Denoiser(waveglow_for_denoiser, mode=denoiser_mode)
    denoiser_strength = 0.005
    # End of parameters

    logging.debug('Tacotron: %s', checkpoint_path)
    logging.debug('Waveglow: %s', waveglow_path)
    logging.debug('AM: SI model')
    logging.debug('is_clip: %d', is_clip)
    logging.debug('Fs: %d', fs)
    logging.debug('Sigma: %f', waveglow_sigma)
    logging.debug('Denoiser strength: %f', denoiser_strength)
    logging.debug('Denoiser mode: %s', denoiser_mode)
    with open(args.hparams,'r') as hpFile:
        hparams = HParamsView(json.load(hpFile))
    taco_stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_acoustic_feat_dims, hparams.sampling_rate,
        hparams.mel_fmin, hparams.mel_fmax)

    # # Load models.
    tacotron_model = load_model(hparams)
    tacotron_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = tacotron_model.eval()
    waveglow_model = load_waveglow_model(waveglow_path)

    if os.path.isfile(teacher_utt_path):
        logging.info('Perform AC on %s', teacher_utt_path)
        #teacher_ppg = get_ppg(teacher_utt_path, deps)
        teacher_ppg = np.load(teacher_utt_path)
        
        gen_mel_filename=os.path.basename(teacher_utt_path).split('.')[0]
        ac_mel = get_inference(teacher_ppg, tacotron_model, is_clip)
        print(gen_mel_filename)
        if args.extract_mel:
            outFilename=os.path.join(output_dir, '{}2mel.pt'.format(gen_mel_filename))
            print(outFilename,ac_mel.cpu()[0].shape)
            torch.save(ac_mel.cpu()[0],outFilename)
            mel=torch.load(outFilename).cpu().detach().numpy()
            
            stft_obj=STFT(filter_length=hparams.filter_length, hop_length=hparams.hop_length, win_length=hparams.win_length)
            griffin_lim(mel,stft_obj)
          
        	

    else:
        logging.warning('Missing %s', teacher_utt_path)

    # logging.info('Done!')
