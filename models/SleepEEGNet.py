import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_1D(nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        drate = 0.5        
        self.RELU = nn.ReLU()          
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.RELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.RELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)        

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        return x_concat
    

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional = True)
        self.device = device

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length,device, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class SleepEEGNet(nn.Module):
    def __init__(self, pretrain = False):
        super(SleepEEGNet, self).__init__()
        drate = 0.5
        self.pretrain = pretrain
        self.feature_extract = CNN_1D()
        self.lstm1 = nn.LSTM(input_size = 10240, hidden_size = 512, num_layers = 2, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size = 1024, hidden_size = 512, num_layers = 2, bidirectional=True)
        self.fc1 = nn.Linear(10240 ,512)
        self.fc2 = nn.Linear(1024+512,5)
        self.fc_pre = nn.Linear(10240,5)
        self.dropout = nn.Dropout(drate)
        #self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.feature_extract(x)
        if self.pretrain:
            x = x.view(x.shape[0],-1)
            return self.fc_pre(x)
        ## LSTM
        x1 = x.view(x.shape[0],1,-1)
        x1, _ = self.lstm1(x1)
        x1 = self.dropout(x1)
        x1, _ = self.lstm2(x1)
        x1 = self.dropout(x1)
        ## FC
        x2 = x.view(x.shape[0],-1)
        x2 = self.fc1(x2)
        ## Concat
        x1 = x1.view(x1.shape[0],-1)
        x_concat = torch.cat((x1, x2), dim=1)
        x_concat = self.dropout(x_concat)
        ## FC and softmax
        
        out = self.fc2(x_concat)        
        return out