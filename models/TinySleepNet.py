import torch
import torch.nn as nn

class TinySleepNet(nn.Module):
    def __init__(self, input_size, num_classes = 5):
        super(TinySleepNet, self).__init__()
        drate = 0.5
        self.num_classes = num_classes
        self.feature_size = 0
        if input_size == 3000:
            self.feature_size = 8192
        elif input_size == 6000:
            self.feature_size = 16256

        self.feature_extract = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(drate)        
        )

        self.lstm = nn.LSTM(input_size = self.feature_size, hidden_size = 128, num_layers = 2, bidirectional=False)
        self.dropout = nn.Dropout(drate)
        self.fc = nn.Linear(128,self.num_classes)        
    
       # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad        

    def forward(self, x):
        ## Feature extraction
        x = self.feature_extract(x)                
        ## LSTM
        x = x.view(x.shape[0],1,-1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        ## FC and softmax       
        x = x.view(x.shape[0],-1) 
        out = self.fc(x)        
        return out
        
if __name__=="__main__":
    model = TinySleepNet(input_size=6000)
    inputs = torch.rand((20,1,6000))
    print(model(inputs))