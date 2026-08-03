import torch
from audio_classifier import AudioClassifier
from training import inference, training
from dataset_loader import SoundsDS
from torch.utils.data import random_split
import pandas as pd
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader

data_path = Path.cwd()/'Acted Emotional Speech Dynamic Database/Acted Emotional Speech Dynamic Database'

metadata_file = data_path/'audio.csv'
df = pd.read_csv(metadata_file)
df.head()

df['relative_path']='/'+ df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)

df = df[['relative_path', 'classID']]
df.head()

myModel = AudioClassifier()
myDs = SoundsDS(df, data_path, False)
num_items = len(myDs)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myDs, [num_train, num_val])

train_dl = DataLoader(train_ds, batch_size = 16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
next(myModel.parameters()).device

statePath = Path.cwd()/'data-xxx.pt'
modelConfig = Path(statePath)

if modelConfig.is_file():
    myModel.load_state_dict(torch.load(statePath, weights_only=True))
    myModel.eval()
else:
    num_epochs = 500
    training(myModel, train_dl, num_epochs, device)
    torch.save(myModel.state_dict(), Path.cwd()/'dataxxx.pt')

inference(myModel, val_dl, device)