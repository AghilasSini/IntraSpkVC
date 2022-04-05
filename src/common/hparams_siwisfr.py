# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA Corporation
# Copyright (c) 2019, Guanlong Zhao
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Modified from https://github.com/NVIDIA/tacotron2"""


class HParamsView(object):
    def __init__(self, d):
        self.__dict__ = d


def create_hparams(**kwargs):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "epochs":500,
        "iters_per_checkpoint":1000,
        "seed": 16807,
        "dynamic_loss_scaling": True,
        "fp16_run": False,
        "distributed_run": False,
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "output_directory": '/gpfsscratch/rech/eyy/uyk62ct/data/data_fr/output/siwis_checkpoints',  # Directory to save checkpoints.
        # Directory to save tensorboard logs. Just keep it like this.
        "log_directory": '/gpfsscratch/rech/eyy/uyk62ct/data/data_fr/log',
        "checkpoint_path": '',  # Path to a checkpoint file.
        "warm_start": False,  # Load the model only (warm start)
        "n_gpus": 1,  # Number of GPUs
        "rank": 0,  # Rank of current gpu
        "group_name": 'group_name',  # Distributed group name

        ################################
        # Data Parameters             #
        ################################
        # Passed as a txt file, see data/filelists/training-set.txt for an
        # example.
        "training_files": '/gpfsscratch/rech/eyy/uyk62ct/data/data_fr/filelists/training_set.txt',
        # Passed as a txt file, see data/filelists/validation-set.txt for an
   
        # example.
        "validation_files": '/gpfsscratch/rech/eyy/uyk62ct/data/data_fr/filelists/validation_set.txt',
        "is_full_ppg": True,  # Whether to use the full PPG or not.
        "is_append_f0": False,  # Currently only effective at sentence level
        "ppg_subsampling_factor": 1,  # Sub-sample the ppg & acoustic sequence.
        # Cases
        # |'load_feats_from_disk'|'is_cache_feats'|Note
        # |True                  |True            |Error
        # |True                  |False           |Please set cache path
        # |False                 |True            |Overwrite the cache path
        # |False                 |False           |Ignores the cache path
        "load_feats_from_disk": False,  # Remember to set the path.
        # Mutually exclusive with 'load_feats_from_disk', will overwrite
        # 'feats_cache_path' if set.
        "is_cache_feats": False,
        "feats_cache_path": '',

        ################################
        # Audio Parameters             #
        ################################
    

        "max_wav_value":32768.0,
        "sampling_rate":22050,
        "filter_length":1024,
        "hop_length":256,
        "win_length":1024,
        "n_mel_channels":80,
        "mel_fmin":0.0,
        "mel_fmax":8000.0,

        ################################
        # Model Parameters             #
        ################################
        "n_symbols": 3536,
        "symbols_embedding_dim": 600,

        # Encoder parameters
        "encoder_kernel_size": 5,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 600,

        # Decoder parameters
        "decoder_rnn_dim": 300,
        "prenet_dim": 300,
        "max_decoder_steps": 1000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,

        # Attention parameters
        "attention_rnn_dim": 300,
        "attention_dim": 150,
        # +- time steps to look at when computing the attention. Set to None
        # to block it.
        "attention_window_size": 20,

        # Location Layer parameters
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,

        # Mel-post processing network parameters
        "postnet_embedding_dim": 512,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate":False,
        "learning_rate":1e-3,
        "weight_decay":1e-6,
        "grad_clip_thresh":1.0,
        "batch_size":64,
        "mask_padding":True  # set model's padded outputs to padded values
        "mel_weight": 1,
        "gate_weight": 0.005
    }

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view


def create_hparams_stage(**kwargs):
    """Create model hyperparameters. Parse nondefault from given string.

    These are the parameters used for our interspeech 2019 submission.
    """

    hparams = {
    ################################
        # Experiment Parameters        #
        ################################
        "epochs" : 500,
        "iters_per_checkpoint" : 1000,
        "seed":1234,
        "dynamic_loss_scaling":True,
        "fp16_run":False,
        "distributed_run":False,
        "dist_backend":"nccl",
        "dist_url":"tcp://localhost:54321",
        "cudnn_enabled":True,
        "cudnn_benchmark":False,
        # ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        "load_mel_from_disk":False,
        "training_files":'',
        "validation_files":'',
        # text_cleaners=['english_cleaners'],
        "checkpoint_path": '',  # Path to a checkpoint file.
        "warm_start": False,  # Load the model only (warm start)
        "n_gpus": 1,  # Number of GPUs
        "rank": 0,  # Rank of current gpu
        "group_name": 'group_name',  # Distributed group name
          "is_full_ppg": True,  # Whether to use the full PPG or not.
        "is_append_f0": False,  # Currently only effective at sentence level
        "ppg_subsampling_factor": 1,  # Sub-sample the ppg & acoustic sequence.
        # Cases
        # |'load_feats_from_disk'|'is_cache_feats'|Note
        # |True                  |True            |Error
        # |True                  |False           |Please set cache path
        # |False                 |True            |Overwrite the cache path
        # |False                 |False           |Ignores the cache path
        "load_feats_from_disk": False,  # Remember to set the path.
        # Mutually exclusive with 'load_feats_from_disk', will overwrite
        # 'feats_cache_path' if set.
        "is_cache_feats": False,
        "feats_cache_path": '',



        ################################
        # Audio Parameters             #
        ################################
        "max_wav_value":32768.0,
        "sampling_rate": 22050,
        "filter_length":1024,
        "hop_length":256,
        "win_length":1024,
        "n_mel_channels":80,
        "mel_fmin":0.0,
        "mel_fmax":8000.0,

        ################################
        # Model Parameters             #
        ################################

        "n_symbols": 3536,
        "symbols_embedding_dim": 600,
        # n_symbols=len(symbols),
        # symbols_embedding_dim=512,

        # Encoder parameters
        "encoder_kernel_size":5,
        "encoder_n_convolutions":3,
        "encoder_embedding_dim":600,

        # Decoder parameters
        "n_frames_per_step":1,  # currently only 1 is supported
        "decoder_rnn_dim":300,
        "prenet_dim":300,
        "max_decoder_steps":1000,
        "gate_threshold":0.5,
        "p_attention_dropout":0.1,
        "p_decoder_dropout":0.1,

        # Attention parameters
        "attention_rnn_dim":300,
        "attention_dim":150,

        # Location Layer parameters
        "attention_location_n_filters":32,
        "attention_location_kernel_size":31,

        # Mel-post processing network parameters
        "postnet_embedding_dim":512,
        "postnet_kernel_size":5,
        "postnet_n_convolutions":5,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate":False,
        "learning_rate":1e-3,
        "weight_decay":1e-6,
        "grad_clip_thresh":1.0,
        "batch_size":64,
        "mask_padding":True,  # set model's padded outputs to padded values

         "mel_weight": 1,
        "gate_weight": 0.005
        }

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view
