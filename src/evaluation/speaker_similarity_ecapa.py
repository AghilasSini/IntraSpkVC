import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
from scipy.spatial.distance import cosine, euclidean
import argparse
import numpy as np


def get_embedding(filepath, model):
    signal, fs = torchaudio.load(filepath)
    embeddings = model.encode_batch(signal)
    
    return embeddings


def compute_cosine_similarity(x, y):
    return 1 - cosine(x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the MuFaSa dataset to give it the Kaldi structure.')
    parser.add_argument('model_dirpath',help='Directory where the speaker verification model is saved')
    parser.add_argument('ref_input_filepath', help='File describing the reference sentences')
    parser.add_argument('synth_waveglow_input_filepath', help='File describing the synthesized sentences')
    parser.add_argument('output_waveglow_filepath', help='Where to save the results of the objective evaluation')
    parser.add_argument('--synth_griffinLim_input_filepath', help='File describing a second (optional) set of synthesized sentences')
    parser.add_argument('--output_griffinLim_filepath', help='Where to save the results of the objective evaluation for the second (optinal) set of synthetic sentences')

    args = parser.parse_args()
    model_dirpath = args.model_dirpath
    ref_input_filepath = args.ref_input_filepath
    synth_waveglow_input_filepath = args.synth_waveglow_input_filepath
    synth_griffinLim_input_filepath = args.synth_griffinLim_input_filepath
    output_waveglow_filepath = args.output_waveglow_filepath
    output_griffinLim_filepath = args.output_griffinLim_filepath

    # model_dirpath = "/vrac/aperquin/test_tts/tal/misc/models/"
    # ref_input_filepath = "/vrac/aperquin/test_tts/tal/misc/ref_files.txt"
    # synth_waveglow_input_filepath = "/vrac/aperquin/test_tts/tal/misc/synth_waveglow_files.txt"
    # # synth_griffinLim_input_filepath = "/vrac/aperquin/test_tts/tal/misc/synth_griffinLim_files.txt"
    # output_waveglow_filepath = "/vrac/aperquin/test_tts/tal/misc/automatic_measure_speaker_waveglow.tsv"
    # # output_griffinLim_filepath = "/vrac/aperquin/test_tts/tal/misc/automatic_measure_speaker_griffinLim.tsv"

    # List the .wav files 
    with open(ref_input_filepath, 'r') as ref_input_file:
        ref_files = [line.strip() for line in ref_input_file.readlines()]
    with open(synth_waveglow_input_filepath, 'r') as synth_waveglow_input_file:
        synth_waveglow_files = [line.strip() for line in synth_waveglow_input_file.readlines()]
    # with open(synth_griffinLim_input_filepath, 'r') as synth_griffinLim_input_file:
    #     synth_griffinLim_files = [line.strip() for line in synth_griffinLim_input_file.readlines()]
    # assert len(ref_files) == len(synth_waveglow_files) == len(synth_griffinLim_files)
    assert len(ref_files) == len(synth_waveglow_files)

    # model = VoiceEncoder()
    speaker_encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=model_dirpath)
    speaker_verification_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=model_dirpath)
    
    # a = [(synth_waveglow_files, open(output_waveglow_filepath, "w")), (synth_griffinLim_files, open(output_griffinLim_filepath, "w"))]
    a = [(synth_waveglow_files, open(output_waveglow_filepath, "w"))]
    for synth_files, output_file in a:
        # Write the header of the output file
        line = "\t".join(["ref_file", 'synth_file', "cosine_similarity", "euclidean_distance", "speaker_verification_pred"])
        output_file.write(line + "\n")

        cosine_similarity_list, euclidean_distance_list, speaker_verification_pred_list = [], [], []
        # For each couple of files, apply the automatic measures
        for i in range(len(ref_files)):
            # if i > 5:
            #     break

            try:
                # Compute speaker similarity
                ref_embed = get_embedding(ref_files[i], speaker_encoder)
                synth_embed = get_embedding(synth_files[i], speaker_encoder)
                cosine_similarity = compute_cosine_similarity(ref_embed, synth_embed)
                euclidean_distance = euclidean(ref_embed, synth_embed)

                # Compute speaker verfication accuracy
                score, prediction = speaker_verification_model.verify_files(ref_files[i], synth_files[i])

                # Write the result for the current couple in the output file
                line = "\t".join([
                    os.path.basename(ref_files[i]), os.path.basename(synth_files[i]),
                    str(cosine_similarity), str(euclidean_distance), str(prediction.cpu().detach().numpy()[0])
                ])
                print(line)
                output_file.write(line + '\n')

                # Store the values for later mean/std computation
                cosine_similarity_list.append(cosine_similarity)
                euclidean_distance_list.append(euclidean_distance)
                speaker_verification_pred_list.append(prediction)
            except Exception as e:
                print(e)

        # Compute the mean and standard deviation or accuracy
        cosine_similarity_mean, cosine_similarity_std = np.mean(cosine_similarity_list), np.std(cosine_similarity_list)
        euclidean_distance_mean, euclidean_distance_std = np.mean(euclidean_distance_list), np.std(euclidean_distance_list)
        speaker_verification_accuracy  = speaker_verification_pred_list.count(True) / len(speaker_verification_pred_list)

        # Write the mean and standard deviation in the file
        line = "\t".join(["mean/accuracy", '', str(cosine_similarity_mean), str(euclidean_distance_mean), str(speaker_verification_accuracy)])
        output_file.write(line + "\n")
        line = "\t".join(["standard deviation", '', str(cosine_similarity_std), str(euclidean_distance_std), ""])
        output_file.write(line + "\n")
