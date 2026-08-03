
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.autograd import Variable
from discriminator import discrmininator
from conditionalGenerator import Generator
from pathlib import Path
from torchvision import datasets
import torchvision.utils as vutils
from PIL import Image
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(device)

path = Path.cwd()/'AxialEmotion'

norm=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
batch_size = 16
image_size = 256
latent_dim = 100
total_labels = 5
num_epochs = 200

transf = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(*norm,inplace=True),
])

def unnorm(images, means, stds):
    means = torch.tensor(means).reshape(1,3,1,1)
    stds = torch.tensor(stds).reshape(1,3,1,1)
    return images*stds+means


dataset = datasets.ImageFolder(root=path,transform=transf)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_audio_vector(batchSize, numClasses, classes):
    target = torch.FloatTensor(batchSize, numClasses).zero_().to(device)
    for classInx in range(len(classes)):
        target[classInx][classes[classInx]] = 1.
    return target

def save_image_set(total_sets, total_images_per_set, currentEpoch):
    modelG.eval()
    with torch.no_grad():
        for i in range(total_sets):
            total_items = total_images_per_set
            for label in range(5): ##label 0 = Anger, 1 = Disgust, 2 = Fear, 3 = Happiness, 4 = Sadness
                noise_vector = torch.randn(total_items, latent_dim, device=device) 
                labels = torch.ones(total_items) * label
                labels = labels.to(device).unsqueeze(1).long()
                oneHotLabels = load_audio_vector(total_items, total_labels, labels)
                fake = unnorm(modelG((noise_vector, oneHotLabels)).detach().cpu(), *norm)
                for k in range(len(fake)):
                    vutils.save_image(fake[k], str(i) + "-" + str(k) + "-label-" + str(label) + "-currentepoch-" + str(currentEpoch) + ".png")
    modelG.train()

modelG = Generator(total_labels)
modelG.to(device)
modelG.apply(weights_init)

modelD = discrmininator(total_labels)
modelD.to(device)
modelD.apply(weights_init)

gStatePath = Path.cwd()/'G-axial-emotion-one-hot-256-lr-1-full-dataset-new-gen-layout-e-500.pth'

modelConfig = Path(gStatePath)

if modelConfig.is_file():
    modelG.load_state_dict(torch.load(gStatePath, weights_only=True))
    save_image_set(10 ,10, 'TestChanges500')



D_optimizer = optim.Adam(modelD.parameters(), lr=0.0001, betas=(0.5, 0.999))
G_optimizer = optim.Adam(modelG.parameters(), lr=0.0001, betas=(0.5, 0.999))
criterion = nn.BCELoss()
total = 0

for epoch in range(1, num_epochs+1): 
    g_loss = 0.0
    d_loss = 0.0
    D_x = 0.0
    D_G_z1 = 0.0
    D_G_z2 = 0.0
    D_loss_list, G_loss_list = [], []
   
    for index, (real_images, labels) in enumerate(dataloader):
        D_optimizer.zero_grad()
        real_images = real_images.to(device)

        b_size = real_images.size(0)
        
        labels = labels.to(device)
        labels = labels.unsqueeze(1).long()

        labs = load_audio_vector(b_size, total_labels, labels)

        real_target = Variable((torch.ones(real_images.size(0), 1)).to(device))
        fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))
        output1 = modelD((real_images, labs))
        output = output1.view(-1)
        D_real_loss = criterion(output1, real_target)
        D_x += output.mean().item()
        # print(discriminator(real_images))
        D_real_loss.backward() # Originally commented - if perf shit comment again
    
        noise_vector = torch.randn(real_images.size(0), latent_dim, device=device)  
        noise_vector = noise_vector.to(device)
        
       
        generated_image = modelG((noise_vector, labs))
        output = modelD((generated_image.detach(), labs))
        D_fake_loss = criterion(output,  fake_target)

    
        # train with fake
        D_fake_loss.backward()
      
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        D_loss_list.append(D_total_loss)
      
        #D_total_loss.backward()
        D_G_z1 += output.view(-1).mean().item()


        errD = D_real_loss + D_fake_loss
        D_optimizer.step()

        # Train generator with real labels
        G_optimizer.zero_grad()
        output = modelD((generated_image, labs))
        G_loss = criterion(output, real_target)
        G_loss_list.append(G_loss)

        G_loss.backward()
        D_G_z2 += output.mean().item()
        G_optimizer.step()

        avg_g_loss = G_loss / total
        G_loss_list.append(avg_g_loss)
        avg_d_loss = D_total_loss / total
        D_loss_list.append(avg_d_loss)

        g_loss += float(G_loss.item())
        d_loss += float(errD.item())
        total += b_size
        
    avg_D_x = D_x / len(dataloader)
    avg_D_G_z1 = D_G_z1 / len(dataloader)
    avg_D_G_z2 = D_G_z2 / len(dataloader)

    print('Epoch: {} \tDiscriminator Loss: {:.10f} \tGenerator Loss: {:.10f} \tD(x): {:.10f} \tD(G(z)): {:.10f} / {:.10f}'.format(
        epoch,
        avg_d_loss,
        avg_g_loss,
        avg_D_x,
        avg_D_G_z1,
        avg_D_G_z2
    ))
    if epoch % 50 == 0:
        save_image_set(1, 5, epoch)
        torch.save(modelG.state_dict(), 'G-axial-emotion-one-hot-256-lr-1-full-dataset-new-gen-layout-e-' + str(epoch) + '.pth')
        torch.save(modelD.state_dict(), 'D-axial-emotion-one-hot-256-lr-1-full-dataset-new-gen-layout-e-' + str(epoch) + '.pth')

print('Finished Training')

save_image_set(1, 10, '256-finished-lr1-new-gen')