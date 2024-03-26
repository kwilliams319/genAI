import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(True),
            nn.Linear(256, 20))
        
        self.decoder = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(True),
            nn.Linear(256, 20))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x