import torch
import torch.nn as nn
import torch.nn.functional as fuc
import torch.optim as opt


class ActorCriticNetwork(nn.Module):
    def __init__(self, in_dims, out_dim, lr):
        super(ActorCriticNetwork, self).__init__()
        fc_dims = [512, 256, 64]
        self.fc1 = nn.Linear(*in_dims, fc_dims[0])
        self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])
        self.fc3 = nn.Linear(fc_dims[1], fc_dims[2])
        self.policy = nn.Linear(fc_dims[-1], out_dim)
        self.value = nn.Linear(fc_dims[-1], 1)
        self.optimizer = opt.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = fuc.relu(self.fc1(state))
        x = fuc.relu(self.fc2(x))
        x = fuc.relu(self.fc3(x))
        policy = self.policy(x)
        value = self.value(x)

        return policy, value

    def save_checkpoint(self, file):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), file)

    def load_checkpoint(self, file):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(file))
