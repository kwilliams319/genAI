import torch
import numpy as np
from model import Autoencoder
import matplotlib.pyplot as plt
import os

# Load the model
model = Autoencoder().cuda()
model.load_state_dict(torch.load('autoencoder.pth'))
model.eval()

# Generate a random sample with zero mean and unit variance at timestep 200
samples = 50
X_ = torch.randn((samples, 20)).cuda()  # Generate random sample for the first 20 elements
# timesteps = torch.arange(200, 0, -1, dtype=torch.float32) / 200  # Generate timestep from 200 to 0
# Concatenate random sample and timestep

# Initialize variables
# 
# Iteratively use the model to predict noise and subtract it
# for t in timesteps:
for t in range(200):
    with torch.no_grad():
        # sample_ = torch.cat([sample, torch.tensor([[t]])], dim=1).cuda()

        noise_ = model(X_)  # Predict noise
        X_ -= noise_  # Subtract predicted noise

        X = X_.reshape(samples, 2, 10).cpu().numpy()
        x, y = X[:, 0].T, X[:, 1].T


        if t%4 == 0:

            plt.cla()
            # plt.axis('equal')
            plt.plot(x, y, 'o')
            plt.plot(np.mean(X[:, 0]), np.mean(X[:, 1]), 'ko')
            plt.plot(0, 0, 'ko')


            th = np.linspace(0, 2*np.pi, 100)[:, None]
            r = 1
            x1, x2 = r*np.cos(th)+ 1.5, r*np.sin(th)+ 1.5
            plt.plot(x1, x2, 'k-')


            # plt.xlim([-1, 3])
            # plt.ylim([-1, 3])
            plt.xlim([-4, 4])
            plt.ylim([-4, 4])
            plt.title(200 - t)
            plt.pause(.01)

            # Save figure
            plt.savefig(os.path.join('figures', f'timestep_{t}.png'))
plt.show()

