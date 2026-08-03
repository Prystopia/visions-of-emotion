import torch.nn as nn
import torch

class discrmininator(nn.Module):
    def __init__(self, num_classes):
        super(discrmininator, self).__init__()
        self.label_condition_disc = nn.Sequential(nn.Linear(num_classes, 3*256*256))
             
        self.model = nn.Sequential(nn.Conv2d(6, 64, 4, 2, 1, bias=False),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(64, 64*2, 4, 3, 2, bias=False),
                     nn.BatchNorm2d(64*2, momentum=0.1,  eps=0.8),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(64*2, 64*4, 4, 3,2, bias=False),
                     nn.BatchNorm2d(64*4, momentum=0.1,  eps=0.8),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Conv2d(64*4, 64*8, 4, 3, 2, bias=False),
                     nn.BatchNorm2d(64*8, momentum=0.1,  eps=0.8),
                     nn.LeakyReLU(0.2, inplace=True), 
                     nn.Conv2d(64*8, 64*16, 4, 3, 2, bias=False),
                     nn.BatchNorm2d(64*16, momentum=0.1,  eps=0.8),
                     nn.LeakyReLU(0.2, inplace=True), 
                     nn.Conv2d(64*16, 64*32, 4, 3, 2, bias=False),
                     nn.BatchNorm2d(64*32, momentum=0.1,  eps=0.8),
                     nn.LeakyReLU(0.2, inplace=True), 
                     nn.Flatten(),
                     nn.Dropout(0.4),
                     nn.Linear(8192, 1),
                     nn.Sigmoid())


    def forward(self, inputs):
        img, label = inputs

        #embed = nn.Linear(5, 5).to("cuda")
        #res = embed(label)

       
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 3, 256, 256)
        concat = torch.cat((img, label_output), dim=1)
        output = self.model(concat)
        return output