from pathlib import Path
import torchvision.transforms as transforms
import torch
import os
import torchvision.utils as vutils
from ConditionalGAN.conditionalGenerator import Generator
import logging

norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
transf = transforms.Compose([
     transforms.Normalize(*norm,inplace=True),
    transforms.Resize(512),
])

def run_art_generation(total_labels, device, audio_output, total_sets, total_images_per_set, folderName):
    statePath = os.path.join(Path.cwd(),'G-axial-emotion-one-hot-256-lr-1-full-dataset-e-300.pth')
    logging.info("State path" + statePath)
    logging.info('Loaded GAN')
    modelConfig = Path(statePath)
    logging.info('Set path:' + statePath)
    logging.info('Load GAN config:' + str(modelConfig.is_file()))
    logging.info("GAN Config size:" + str(os.path.getsize(modelConfig)))

    if not os.path.exists(folderName):
        os.makedirs(folderName)
    
    modelG = Generator(total_labels)
    modelG.to(device)
    modelG.load_state_dict(torch.load(statePath, map_location=torch.device('cpu'), weights_only=True))
    modelG.eval()
    logging.info('Evaluated GAN Weights')
    with torch.no_grad():
        for i in range(total_sets):
            total_items = total_images_per_set
            noise_vector = torch.randn(total_items, 100, device=device)
            oneHotLabels = load_audio_vector(total_items, total_labels, audio_output, device)
            fake = unnorm(modelG((noise_vector, oneHotLabels)).detach().cpu(), *norm)
            for k in range(len(fake)):
                vutils.save_image(transf(fake[k]), os.path.join(folderName, str(i) + "-" + str(k) + "-live.png"))
            #Probably what I need to do here is to save the images to memory to them feed into the resolution GAN - for this stage this is ok

def unnorm(images, means, stds):
    means = torch.tensor(means).reshape(1,3,1,1)
    stds = torch.tensor(stds).reshape(1,3,1,1)
    return images*stds+means

def load_audio_vector(batchSize, numClasses, audio_output, device):
    target = torch.FloatTensor(batchSize, numClasses).zero_().to(device)
    for classInx in range(batchSize):
        for labelVal in range(len(audio_output)):
            target[classInx][labelVal] = audio_output[labelVal]
    return target