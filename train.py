import torch
from gymnasium.wrappers import FlattenObservation

from core.environment import Parkour
from core.model import JumperBot, train

env = FlattenObservation(Parkour(render_mode="human"))
while True:
    model = JumperBot()

    train(
        model,
        env,
        2000,
        2000,
        0.99
    )

    torch.save(
        model.network.state_dict(),
        r"models\checkpoint.pth"
    )
    print("saved and resetting")

env.close()