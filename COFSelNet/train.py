import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from COFSelNetnet import MyModel
#Training COFSelNet network



batch_size=256
device = torch.device('cuda:0' if torch.cuda.is_available() else 'CPU')

branch1_data = torch.load('branch1max.pth')
branch2_data = torch.load('branch2max.pth')
labels = torch.load('labelsmax.pth')

# branch1_data,branch2_data,labels=branch1_data[0:50000,:,:],branch2_data[0:50000,:,:],labels[0:50000,:]
vgg = MyModel().to(device)
criterion =nn.L1Loss().to(device)
optimizer = optim.Adam(vgg.parameters(), lr=0.001)

train_dataset = TensorDataset(torch.tensor(branch1_data, dtype=torch.float32),
                              torch.tensor(branch2_data, dtype=torch.float32),
                              torch.tensor(labels, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


# Training loop
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
num_epochs = 100
for epoch in range(num_epochs):
    
    for batch in train_loader:
        branch1_input,branch2_input, target = batch
        branch1_input = branch1_input.to(device)
        branch2_input = branch2_input.to(device)
        target = target.to(device)

        # Forward pass
        output = vgg(branch1_input,branch2_input)

        # Squeeze the target to match the output shape
        target = target.squeeze()

        # Calculate loss
        loss = criterion(output.squeeze(), target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Learning Rate: {scheduler.get_last_lr()}")
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    torch.save(vgg.state_dict(), 'COFSelNet_weight2.pth')

