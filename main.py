import os
import pickle
import random
import argparse
from glob import glob

import numpy as np
import pandas as pd

import wandb
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

import utils.loss as loss
from utils.train import fit, test
from utils.util import calc_class_weight
from utils.dataloader import data_generator_np
from models.AttnSleep import AttnSleep
from models.DeepSleepNet import DeepSleepNet
from models.TinySleepNet import TinySleepNet



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def main(args):        
    set_seed(42)
    ## Initial params
    DATA_PATH = args.data_path    
    ROOT_PATH = os.getcwd()
    PROJECT_WANDB = "SLEEP_TEMPLETE_2nd"
    RUN_NAME = args.save_path    
    BATCH_SIZE = args.batch_size
    SAVE_PATH = ROOT_PATH + f'/saved_model/{RUN_NAME}'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)        
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f'CUDA is available. Your device is {DEVICE}.')
    else:
        print(f'CUDA is not available. Your device is {DEVICE}. It can take long time training in CPU.')    
    NUM_CHANNELS = args.channel



    ## DataPath
    list_files = sorted(glob(os.path.join(DATA_PATH,"*.npz")))    
    list_name = np.unique([x[-9:-6] for x in list_files])    
    train_valid_list = list_name[:-5]
    test_list = list_name[-5:] 
    ## Fold    
    n_folds = 10
    splits = KFold(n_splits = n_folds, shuffle = True, random_state = 42)

    ## K-Fold validation
    results = dict()      
    all_pred = []
    all_true = []          
    for fold_, (train_idx, valid_idx) in enumerate(splits.split(train_valid_list)):
        print("fold n°{}".format(fold_+1))
        modelPath = os.path.join(SAVE_PATH, f'bestModel_fold{fold_+1}.pt')
        wandb.init(project = PROJECT_WANDB, config = args, name = RUN_NAME + f'_fold{fold_+1}')
        ## Split dataset
        train_list = [x for x in list_files if x[-9:-6] in train_valid_list[train_idx]]
        valid_list = [x for x in list_files if x[-9:-6] in train_valid_list[valid_idx]]    
        train_loader, valid_loader, counts = data_generator_np(train_list, valid_list, BATCH_SIZE, num_classes = NUM_CHANNELS, augmentation=args.augmentation)    
        weights_for_each_class = calc_class_weight(counts)
        ## Model
        if args.model == "TinySleepNet":
            model = TinySleepNet(input_size = 3000, num_classes = args.channel)
        elif args.model == "DeepSleepNet":
            model = DeepSleepNet(num_classes= args.channel)
        else:
            model = AttnSleep(num_classes= args.channel)            
        model.apply(weights_init_normal)
        wandb.watch(model, log="all")
        # train/validate now
        criterion = getattr(loss, "weighted_CrossEntropyLoss")
        optimizer = torch.optim.Adam(params = model.parameters(),lr = args.initial_lr, weight_decay= args.initial_lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)                        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 5)  
        history = fit(epochs = args.epochs,
                      model = model,
                      train_loader = train_loader,
                      val_loader = valid_loader,
                      criterion = criterion, 
                      optimizer = optimizer, 
                      path = modelPath, 
                      class_weights = weights_for_each_class,
                      scheduler= scheduler,
                      earlystop = 20,                      
                      device = DEVICE)
        results[fold_+1] = history['val_acc'][-1]
        with open(os.path.join(SAVE_PATH, f"hist_fold{fold_+1}.pkl"), "wb") as file:
            pickle.dump(history, file)
        wandb.finish()

        ## Valid check for classification report
        model.load_state_dict(torch.load(modelPath))
        yPred, yTrue = test(model,valid_loader, DEVICE)
        all_pred.extend(yPred)
        all_true.extend(yTrue)


    # Print fold results
    print(f'\nK-FOLD CROSS VALIDATION RESULTS FOR {n_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value*100} %')
        sum += value
    print(f'Average: {sum/len(results.items())*100} %')
    
    ## Save classification reports
    r = classification_report(all_true, all_pred, digits=6, output_dict=True)
    cm = confusion_matrix(all_true, all_pred)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(all_true, all_pred)
    df["accuracy"] = accuracy_score(all_true, all_pred)
    df = df * 100
    df.to_excel(os.path.join(SAVE_PATH, f"{args.save_path}_classification_report.xlsx"))
    torch.save(cm, os.path.join(SAVE_PATH, f"{args.save_path}_confusion_matrix.torch"))    


if __name__ == "__main__":
    
    ## Argparse
    parser = argparse.ArgumentParser(description='Sleep_python', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Fix setting
    parser.add_argument('--epochs', default=200, type=int,
                        help='Epochs')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch Size')

    ## Changable settings                     
    parser.add_argument('--data_path', default="../", type=str,
                        help='Set source(train) path')
    parser.add_argument('--model', default='TinySleepNet', type=str,choices={"TinySleepNet", "DeepSleepNet","AttnSleep"},
                        help='Model')    
    parser.add_argument('--save_path', default='..', type=str,
                        help='Set save path')
    parser.add_argument('--initial_lr', default=1e-4, type=float,
                        help='Set initial learning rate')
    parser.add_argument('--channel', default=5, type=int,
                        help='Augmentation')        
    parser.add_argument('--augmentation', default=None, type=str,
                        help='Augmentation')    
    args = parser.parse_args()

    main(args)
