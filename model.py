import torch 
import torch.nn as nn 
import torch.nn.functional as F 




class Network(nn.Module):
    def __init__(self):
        super().__init__() 

        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True)

        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)
        self.fc7 = nn.Linear(16, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        hidden_state, _ = self.lstm(x)

        feature = hidden_state[:, 0, :]

        feature = F.relu(self.fc4(feature))
        feature = F.relu(self.fc5(feature))
        delta_x = self.fc6(feature) 
        delta_y = self.fc7(feature)

        return delta_x, delta_y 
    




