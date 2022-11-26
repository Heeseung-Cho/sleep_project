import torch
import torch.nn as nn

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
    
class DeepSleepNet(nn.Module):
    def __init__(self, pretrain = False, num_classes = 5):
        super(DeepSleepNet, self).__init__()
        drate = 0.5
        self.pretrain = pretrain
        self.num_classes = num_classes
        self.feature_extract = CNN_1D()
        self.lstm1 = nn.LSTM(input_size = 10240, hidden_size = 512, num_layers = 2, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size = 1024, hidden_size = 512, num_layers = 2, bidirectional=True)
        self.fc1 = nn.Linear(10240 ,512)
        self.fc2 = nn.Linear(1024+512,self.num_classes)
        self.fc_pre = nn.Linear(10240,self.num_classes)
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