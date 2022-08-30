import os
import torchaudio
from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
from scipy.spatial.distance import cosine


def get_embedding(filepath, model):
    signal, fs = torchaudio.load(filepath)
    embeddings = model.encode_batch(signal)
    
    return embeddings


def compute_cosine_similarity(x, y):
    return 1 - cosine(x, y)


if __name__ == "__main__":
    model_dirpath = "/vrac/aperquin/test_tts/tal/misc/models/"
    ref_input_filepath = "/vrac/aperquin/test_tts/tal/misc/ref_files.txt"
    synth_waveglow_input_filepath = "/vrac/aperquin/test_tts/tal/misc/synth_waveglow_files.txt"
    # synth_griffinLim_input_filepath = "/vrac/aperquin/test_tts/tal/misc/synth_griffinLim_files.txt"
    output_waveglow_filepath = "/vrac/aperquin/test_tts/tal/misc/automatic_measure_speaker_waveglow.tsv"
    # output_griffinLim_filepath = "/vrac/aperquin/test_tts/tal/misc/automatic_measure_speaker_griffinLim.tsv"

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
        line = "\t".join(["ref_file", 'synth_file', "cosine_similarity", "euclidean_similarity", "speaker_verification_pred"])
        output_file.write(line + "\n")

        # For each couple of files, apply the automatic measures
        for i in range(len(ref_files)):
            # if i > 5:
            #     break

            # Compute speaker similarity
            ref_embed = get_embedding(ref_files[i], speaker_encoder)
            synth_embed = get_embedding(synth_files[i], speaker_encoder)
            cosine_similarity = compute_cosine_similarity(ref_embed, synth_embed)

            # Compute speaker verfication accuracy
            score, prediction = speaker_verification_model.verify_files(ref_files[i], synth_files[i])

            # Write the result for the current couple in the output file
            line = "\t".join([
                os.path.basename(ref_files[i]), os.path.basename(synth_files[i]),
                str(cosine_similarity), str(score)
            ])
            output_file.write(line + '\n')

