import pathlib
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class JumperBot(nn.Module):
    input_size = 262
    output_size = 2

    def __init__(self, *, model: nn.Sequential | pathlib.Path | None = None, epsilon: float = 0.5, lr: float = 0.0001) -> None:
        super().__init__()

        if model is None or isinstance(model, pathlib.Path): # Create a new model
            self.network = nn.Sequential(
                nn.Linear(self.input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, self.output_size),
                nn.Tanh()
            )

            if isinstance(model, pathlib.Path):
                if model.is_file() and model.suffix == ".pth":
                    self.network.load_state_dict(torch.load(model))
                else:
                    raise FileNotFoundError("No valid PyTorch model file found at the given path")

        else: # Load an already trained model (with nn.Sequential typing)
            input_layer = model[0]
            output_layer = model[-1]

            if (
                input_layer.in_features == self.input_size
                and output_layer.out_features == self.output_size
            ):
                self.network = model
            else:
                raise ValueError("Provided model has incompatible dimensions")

        self.epsilon = epsilon
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=lr
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class ReplayMemory:
    def __init__(self, capacity=15000) -> None:
        self.memory = deque(maxlen=capacity)

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float) -> None:
        self.memory.append(
            (state, action, reward)
        )

    def sample(self, batch_size: int) -> list[tuple[dict, torch.Tensor, float]]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

def train(
    model: JumperBot, 
    env, 
    episodes=100,
    steps=2000,
    gamma=0.99
) -> None:
    memory = ReplayMemory(30000)

    for episode in range(episodes):

        obs, info = env.reset()

        log_probs = []
        rewards = []

        for _ in range(steps):

            obs_tensor = torch.from_numpy(obs).float()

            mean_action = model(obs_tensor)

            noise = torch.randn_like(mean_action) * 0.2
            action = torch.clamp(mean_action + noise, -1, 1)

            next_obs, reward, terminated, truncated, info = env.step(
                action.detach().numpy()
            )

            if reward > 0:
                memory.add(
                    obs_tensor,
                    action.detach(),
                    reward
                )

            log_prob = -((action - mean_action) ** 2).sum()

            log_probs.append(log_prob)
            rewards.append(reward)

            obs = next_obs

            if terminated or truncated:
                break

        returns = []
        total = 0

        for reward in reversed(rewards):
            total = reward + gamma * total
            returns.insert(0, total)

        returns = torch.tensor(returns)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = torch.tensor(0.0)

        for log_prob, R in zip(log_probs, returns):
            policy_loss -= log_prob * R

        replay_loss = torch.tensor(0.0)

        if len(memory) > 32:
            batch = memory.sample(32)

            for state, action, reward in batch:
                prediction = model(state)
                replay_loss += ((prediction - action) ** 2).mean()

            replay_loss /= 32

        loss = policy_loss + 0.1 * replay_loss

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        print(
            f"Episode {episode}: "
            f"reward={sum(rewards):.2f}, "
            f"policy_loss={policy_loss.item():.3f}, "
            f"replay_loss={replay_loss.item() if len(memory) > 32 else 0:.3f}"
        )