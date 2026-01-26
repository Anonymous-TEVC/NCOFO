
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.maxpool = nn.MaxPool1d(1024)
        self.fc4 = nn.Linear(2048, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, 1)
        self.fc8 = nn.Linear(2, 64)
        self.relu1 = nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def forward(self, x1,x2):
        # x: (batch_size, n, 3)
        x1 = self.fc1(x1)
        x1=self.relu1(x1)
        x1 = self.fc2(x1)
        x1=self.relu1(x1)
        x1 = self.fc3(x1)
        x1=self.relu1(x1)
        # x: (batch_size, n, 1024)
        x1, _ = torch.max(x1, dim=1, keepdim=True)
        # x: (batch_size, 1, 1024)
        x1 = x1.view(-1, 1024)
        # x: (batch_size, 1024)

        x2 = self.fc8(x2)
        x2=self.relu1(x2)
        x2 = self.fc2(x2)
        x2=self.relu1(x2)
        x2 = self.fc3(x2)
        x2=self.relu1(x2)
        # x: (batch_size, n, 1024)
        x2, _ = torch.max(x2, dim=1, keepdim=True)
        # x: (batch_size, 1, 1024)
        x2 = x2.view(-1, 1024)

        x = torch.cat((x1 , x2), dim=1)
        x = self.fc4(x)
        x=self.relu1(x)
        x = self.fc5(x)
        x=self.relu1(x)
        x = self.fc6(x)
        x=self.relu1(x)
        x = self.fc7(x)
        x=self.sigmoid(x)
        return x
