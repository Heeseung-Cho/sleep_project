import torch
from torch.utils.data import Dataset
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset, num_classes = 5, augmentation = None):
        super(LoadDataset_from_numpy, self).__init__()

        # load files
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])

        ## Classes
        if num_classes == 4:
            y_train = np.array([convert4class(x) for x in y_train])
        elif num_classes == 3:
            y_train = np.array([convert3class(x) for x in y_train])        
        
        ## Augmentation        
        if augmentation == "SMOTE":
            X_train, y_train = SMOTE().fit_resample(X_train.squeeze(), y_train)            
        elif augmentation == "TLink":
            X_train, y_train = TomekLinks().fit_resample(X_train.squeeze(), y_train)            
        elif augmentation == "SMOTE+TLink":
            X_train, y_train = SMOTETomek().fit_resample(X_train.squeeze(), y_train)            
        else:
            pass
        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()
        self.num_classes = num_classes

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        bin_labels = np.bincount(self.y_data)
                                    
        print(f"Labels count: {bin_labels}")
        print(f"Shape of Input : {self.x_data.shape}")
        print(f"Shape of Labels : {self.y_data.shape}")

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def data_generator_np(training_files, subject_files, batch_size, num_classes = 5, augmentation = None):
    train_dataset = LoadDataset_from_numpy(training_files, num_classes = num_classes, augmentation = augmentation)
    test_dataset = LoadDataset_from_numpy(subject_files, num_classes = num_classes, augmentation = None)

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts


def convert3class(y):
    if y == 1:
        return 1
    elif y == 2:
        return 1
    elif y == 3:
        return 2
    elif y == 4:
        return 2
    else: ## 'REM'
        return 0

def convert4class(y):
    if y == 4:
        return 3
    else:    
        return y

if __name__=="__main__":
    sample = ['../../Sleep/dataset/Nasal_3000dim/sleep_stage_go002-1.npz', '../../Sleep/dataset/Nasal_3000dim/sleep_stage_go002-2.npz',]
    LoadDataset_from_numpy(sample, augmentation="SMOTE+TLink")