import pandas
import sys
import os
from collections import Counter

import numpy as np


from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt

from tqdm import tqdm



#  Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = torchaudio.transforms.MFCC(sample_rate=22050)

transform = transform.to(device)   



learning_rate = 0.01









#defining dataset class
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = len(self.x)

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  

    def __len__(self):
        return self.length


def label_to_index(emotion):
    # Return the position of the word in labels
    return torch.tensor(labels.index(emotion))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]



def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0,1)



def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for data,label in batch:
        tensors += [data]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)


    return tensors, targets







import argparse
import pandas as pd


# Raise this exception if a file is not valid
class FileNotExistException(Exception):
    def __init__(self,file_path,message="This file does not exists"):
        self.file_path = file_path
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return f'{self.file_path} -> {self.message}'


class HParamsView(object):
    def __init__(self, d):
        self.__dict__ = d

# ######################################################################
# # Training and Testing the Network
# # --------------------------------
# #
# # Now let’s define a training function that will feed our training data
# # into the model and perform the backward pass and optimization steps. For
# # training, the loss we will use is the negative log-likelihood. The
# # network will then be tested after each epoch to see how the accuracy
# # varies during the training.
# # #



def train(model,train_loader, epoch,pbar,hparams):
    model.train()
    losses=[]
    hparams=HParamsView(hparams)
    for batch_idx,(x_train,y_train) in enumerate(train_loader):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        # apply transform and model on whole batch directly on device
        data = transform(x_train)
        output = model(data)

        #calculate loss
        loss = hparams.loss_fn(output,y_train.reshape(-1,1,1))
        # predicted = model(torch.tensor(data,dtype=torch.float32).to(device))

        # acc = (predicted.reshape(-1).detach().numpy().round() == y_train).mean()    #backprop

        hparams.optimizer.zero_grad()
        loss.backward()
        hparams.optimizer.step()
       


       
        # print training stats
        if batch_idx % hparams.log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(hparams.pbar_update)
        # record loss
        losses.append(loss.item())


def test(model,test_loader, epoch,pbar,hparams):
        model.eval()
        hparams=HParamsView(hparams)
        correct = 0
        for data, target in test_loader:

            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = transform(data)
           
            predicted = model(torch.tensor(data,dtype=torch.float32).to(device))
            correct += number_of_correct(predicted.squeeze().detach(), target)

            # update progress bar
            pbar.update(hparams.pbar_update)

        print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")


def number_of_correct(pred, target):
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

# Parse/Manage the arguments
def parser_build():
    parser = argparse.ArgumentParser(description="Sentence Spliter For French")
    parser.add_argument('dataset', type=str, help='data set formating as .tsv')
    parser.add_argument('input_dir', type=str, help='input directory ')

    return parser.parse_args()


  
  


def main():
    args=parser_build()

    dataFilename =args.dataset #"/vrac/tmp/sini/Corpora/Multimodal_classifier/emotion_lab.tsv"
    indir_fpath=args.input_dir #"/vrac/tmp/sini/Corpora/Multimodal_classifier/synpaflex_emo_class"
    


    batch_size=128
    df=pandas.read_csv(dataFilename,sep='|')
    X=[ torchaudio.load(wf)[0][0] for wf in  df['clips_with_full_path'].to_list()] 
    y=df['label'].to_list()
    # for index, row in df.iterrows():
    #     clabel=row['label'] #.split('_')
    #     if clabel=="neutral":
    #         plabel=0
    #     else:
    #         plabel=1

    #     y.append(plabel)

    #     try:
    #         fpath_clip=os.path.join(indir_fpath,row['clip_id'].strip()+'.wav')
    #         print("clip_id {}  ".format(fpath_clip))
    #         if not os.path.exists(fpath_clip):
    #             raise FileNotExistException(fpath_clip)
    #         waveform,_=torchaudio.load(fpath_clip)
    #         X.append(waveform[0])
    #     except FileNotExistException as e:
    #         print(e)

    assert len(y) == len(X) 
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


  


   

   




    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False



# loading  train set
    trainset = dataset(X_train,y_train)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

# loading  validation set
    validset = dataset(X_val,y_val)
    val_loader = torch.utils.data.DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

# loading test set
    testset = dataset(X_test,y_test)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
       collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )



    transformed=transform(X[0].to(device))

    from conv_net import M5

    model = M5(n_input=transformed.shape[0], n_output=1)
    model.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    n = count_parameters(model)
    print("Number of parameters: %s" % n)


        ######################################################################
    # We will use the same optimization technique used in the paper, an Adam
    # optimizer with weight decay set to 0.0001. At first, we will train with
    # a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
    # to 0.001 during training after 20 epochs.
    #


    pbar_update = 1 / (len(train_loader) + len(test_loader))
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

    n_epoch=70
    log_interval=64

    hparams_dict={
        "log_interval":log_interval,
        "optimizer":optimizer,

        "loss_fn" :loss_fn,
        "pbar_update":pbar_update,


    }
    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            train(model,train_loader, epoch, pbar, hparams_dict)
            test(model,test_loader, epoch, pbar, hparams_dict)
            scheduler.step()




  




    

    # ######################################################################
    # # Finally, we can train and test the network. We will train the network
    # # for ten epochs then reduce the learn rate and train for ten more epochs.
    # # The network will be tested after each epoch to see how the accuracy
    # # varies during the training.
    # #

    # log_interval = 20
    # n_epoch = 1

    # pbar_update = 1 / (len(train_loader) + len(test_loader))
    # losses = []

    # # The transform needs to live on the same device as the model and the data.
    # transform = transform.to(device)
    # with tqdm(total=n_epoch) as pbar:
    #     for epoch in range(1, n_epoch + 1):
    #         train(model, epoch, log_interval)
    #         test(model, epoch)
    #         scheduler.step()

    # y_true=[]
    # y_pred=[]
    # for batch_idx, (data, target) in enumerate(test_loader):
    #     data = data.to(device)
    #     target = target.to(device)
    #     # y_true.append(target)
    #     # apply transform and model on whole batch directly on device
    #     data = transform(data)
    #     output = model(data)

    #     pred = get_likely_index(output)

    #     for i, x in enumerate(pred.numpy()):
    #         y_pred.append(x[0])
    #         print(x)
    #     # y_true+=target.t().numpy()
    #     for i, x in enumerate(target.numpy()):
    #         y_true.append(x)
        
    # print(y_true,y_pred)
    # from sklearn.metrics import classification_report

    # target_names = ['class 0', 'class 1']
    # print(classification_report(y_true, y_pred, target_names=target_names))

if __name__ == '__main__':
    main()





# with open(file) as f:
#     reader = csv.reader(f, delimiter="|")
#     label_col = list(zip(*reader))[0]
#     print(label_col)


