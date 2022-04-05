import sys
import os
from random import choice
import librosa

def file_list (folder_path, max_len, min_len=0):
    _, _, filenames = next(os.walk(folder_path))
    res = []
    for f in filenames:
        duration = librosa.get_duration(filename=os.path.abspath(os.path.join(folder_path, f)))
        if duration < max_len and duration > min_len:
            res.append(f)
    return res

def generate_training_validation_txt (folder_path, filelists_dest):
    filenames = file_list(folder_path, 15)

    train, test = split(filenames, 0.6)
    
    prefix_data = os.path.abspath(folder_path)
    prefix_filelists = os.path.abspath(filelists_dest)
    
    with open(os.path.join(prefix_filelists, "training_set.txt"), "w") as train_file:
        for elt in train:
            train_file.write(os.path.join(prefix_data, elt) + "\n") 
    
    with open(os.path.join(prefix_filelists, "validation_set.txt"), "w") as test_file:
        for elt in test:
            test_file.write(os.path.join(prefix_data, elt) + "\n") 
   
def generate_spk2utt (folder_path, spk, dest):
    filenames = file_list(folder_path, 15)
    with open(dest, "w") as s2u_file:
        s2u_file.write(spk + " ")
        for elt in filenames:
            name = elt.split(".wav")[0]
            s2u_file.write(name + " ")
    
def generate_wav_scp (folder_path, dest):
    filenames = file_list(folder_path, 15)
    with open(dest, "w") as wav_file:
        for elt in filenames:
            name = elt.split(".wav")[0]
            wav_file.write(name + " " + os.path.abspath(os.path.join(folder_path, elt)) + "\n")

def split(arr, prop):
    limit = int(len(arr) * prop)
    print(len(arr))
    res = []
    while len(res) < limit:
        elt = choice(arr)
        res.append(elt)
        arr.remove(elt)
    
    print(res)
    print(arr)
    return (res, arr)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Expected 2 arguments : gen_train_val_txt.py WAV_FILES_FOLDER OUTPUT_FOLDER [SPK_NAME]")
    if len(sys.argv) == 3:
        name = "Unnamed_spk"
    else:
        name = sys.argv[3]
    generate_training_validation_txt(sys.argv[1], sys.argv[2])
    generate_wav_scp (sys.argv[1], sys.argv[2] + "/wav.scp")
    generate_spk2utt(sys.argv[1], name, sys.argv[2] + "/spk2utt")
