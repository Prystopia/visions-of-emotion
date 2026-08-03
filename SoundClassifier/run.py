import logging
from pathlib import Path
import torch
from SoundClassifier.dataset_loader import SoundsDS
from SoundClassifier.audio_classifier import AudioClassifier
from torch.utils.data import DataLoader
import pandas as pd
import os

def run_audio_classifier(path_to_sound_file, device):
    logging.info('Loading sound classifier')

    statePath = os.path.join(Path.cwd(),'dataxxx.pt')
    logging.info('Loaded sound classifier')
    modelConfig = Path(statePath)
    logging.info('Set path:' + statePath)
    myModel = AudioClassifier()
    logging.info('Load config:' + str(modelConfig.is_file()))
    logging.info("Config size:" + str(os.path.getsize(modelConfig)))

    if modelConfig.is_file():
        myModel.load_state_dict(torch.load(statePath, map_location=torch.device('cpu'), weights_only=True))
        myModel.eval()
    
    logging.info('Setting up classifier to device')
    myModel = myModel.to(device)
    zero = pd.DataFrame()
    myDs = SoundsDS(zero, path_to_sound_file, True)
    logging.info('Initialised SoundsDS ')
    dl = DataLoader(myDs, batch_size = 1)
    for i, data in enumerate(dl):
        inputs = data[0].to(device)
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
        outputs = myModel(inputs)
        _,prediction = torch.max(outputs, 1)
    logging.info('Completed emotion detection')
    return outputs.tolist()[0]

