from networks import ActorCriticNetwork
import torch
import torch.nn.functional as fuc


class Agent(object):
    def __init__(self, in_dims, out_dim, lr=0.000005, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        self.net = ActorCriticNetwork(
            in_dims, out_dim,
            self.lr
        )
        self.log_prob = None

    def get_action(self, observation):
        tensor = torch.tensor([observation], dtype=torch.float)
        state = tensor.to(self.net.device)
        policy, _ = self.net.forward(state)
        policy = fuc.softmax(policy, dim=1)
        probabilities = torch.distributions.Categorical(policy)
        action = probabilities.sample()
        self.log_prob = probabilities.log_prob(action)
        return action.item()

    def learn(self, state, reward, next_state, done):
        self.net.optimizer.zero_grad()

        tensor = torch.tensor([state], dtype=torch.float)
        state = tensor.to(self.net.device)
        tensor = torch.tensor([next_state], dtype=torch.float)
        next_state = tensor.to(self.net.device)
        tensor = torch.tensor([reward], dtype=torch.float)
        reward = tensor.to(self.net.device)

        _, value = self.net.forward(state)
        _, next_value = self.net.forward(next_state)

        delta = reward + self.gamma * next_value * (1 - int(done)) - value

        actor_loss = -self.log_prob * delta
        critic_loss = delta ** 2

        (actor_loss + critic_loss).backward()
        self.net.optimizer.step()
