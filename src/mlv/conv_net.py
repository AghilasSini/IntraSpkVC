######################################################################
# Define the Network
# ------------------
#
# For this tutorial we will use a convolutional neural network to process
# the raw audio data. Usually more advanced transforms are applied to the
# audio data, however CNNs can be used to accurately process the raw data.
# The specific architecture is modeled after the M5 network architecture
# described in `this paper <https://arxiv.org/pdf/1610.00087.pdf>`__. An
# important aspect of models processing raw audio data is the receptive
# field of their first layer’s filters. Our model’s first filter is length
# 80 so when processing audio sampled at 8kHz the receptive field is
# around 10ms (and at 4kHz, around 20 ms). This size is similar to speech
# processing applications that often use receptive fields ranging from
# 20ms to 40ms.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




class M5(nn.Module):

    def __init__(self, num_layers=1, n_input=40, n_output=35, stride=16, n_channel=32,padding=2):
        super(M5, self).__init__()
        
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers

        self.conv1 = torch.nn.Conv1d(n_input, n_channel, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = torch.nn.BatchNorm1d(n_channel)
        self.pool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(n_channel, n_channel, kernel_size=2, padding=padding)
        self.bn2 = torch.nn.BatchNorm1d(n_channel)
        self.pool2 = torch.nn.MaxPool1d(2)
        self.conv3 = torch.nn.Conv1d(n_channel, 2 * n_channel, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm1d(2 * n_channel)
        self.pool3 = torch.nn.MaxPool1d(2)
        self.conv4 = torch.nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=1)
        self.bn4 = torch.nn.BatchNorm1d(2 * n_channel)
        self.pool4 = torch.nn.MaxPool1d(1)
        self.fc1 = torch.nn.Linear(2 * n_channel, n_output)

        # self.relu = nn.ReLU()



        
    def forward(self, x):
        # x_b=x
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return  torch.round(torch.sigmoid(x))
  #F.log_softmax(x, dim=2)  # {'raw_x':x_b,'fc1' : F.log_softmax(x, dim=2),"conv1":x0,"conv2":x1,"conv3": x2,"conv4": x3}

        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        #c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        #output, (hn, cn) = self.lstm(x) #lstm with input, hidden, and internal state
        #hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        #out = self.relu(hn)
        #out = self.fc_1(out) #first Dense
        #out = self.relu(out) #relu
        #out = self.fc(out) #Final Output

        return F.log_softmax(x, dim=2) #{'raw_x':x_b,'fc1' : F.log_softmax(x, dim=2),"conv1":x0,"conv2":x1,"conv3": x2,"conv4": x3}
