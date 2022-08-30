import os
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine


def get_embedding(filepath, model):
    ref_wav = preprocess_wav(filepath)
    ref_embed = model.embed_utterance(ref_wav)
    
    return ref_embed


def compute_cosine_similarity(x, y):
    return 1 - cosine(x, y)


if __name__ == "__main__":
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

    encoder = VoiceEncoder()
    # a = [(synth_waveglow_files, open(output_waveglow_filepath, "w")), (synth_griffinLim_files, open(output_griffinLim_filepath, "w"))]
    a = [(synth_waveglow_files, open(output_waveglow_filepath, "w"))]
    for synth_files, output_file in a:
        # Write the header of the output file
        line = "\t".join(["ref_file", 'synth_file', "cosine_similarity", "euclidean_similarity", "verification_accuracy"])
        output_file.write(line + "\n")

        # For each couple of files, apply the automatic measures
        for i in range(len(ref_files)):
            # if i > 5:
            #     break

            ref_embed = get_embedding(ref_files[i], encoder)
            synth_embed = get_embedding(synth_files[i], encoder)
            cosine_similarity = compute_cosine_similarity(ref_embed, synth_embed)

            # Write the result for the current couple in the output file
            line = "\t".join([
                os.path.basename(ref_files[i]), os.path.basename(synth_files[i]),
                str(cosine_similarity)
            ])
            output_file.write(line + '\n')

