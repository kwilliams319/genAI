import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import Autoencoder  # Import the Autoencoder class from model.py
from itertools import count
from data import gen_data


# Define the model and move to CUDA if available
model = Autoencoder().cuda()
model.load_state_dict(torch.load('autoencoder.pth'))
model.train()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

while True:
    gen_data()
    # Load the training data and convert to PyTorch tensors
    X_ = torch.tensor(np.load('X_.npy'), dtype=torch.float32)
    N = torch.tensor(np.load('N.npy'), dtype=torch.float32)
    # Create a DataLoader for training
    train_loader = DataLoader(TensorDataset(X_, N), batch_size=64, shuffle=True)

    # Training loop
    # for epoch in count(0):  # Training loop
    for epoch in range(1): 
        
        epoch_loss = 0.0
        for x_, n in train_loader:
            x_ = x_.cuda()  # Move batch to CUDA
            n = n.cuda()

            noise_ = model(x_)
            loss = torch.mean((noise_ - n) ** 2)  # Calculate MSE loss manually

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()# * x_.size(0)

        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}], Loss: {epoch_loss:.5f}")

        # Save the trained model
        torch.save(model.state_dict(), 'autoencoder.pth')
