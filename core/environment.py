import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Callable
from enum import IntEnum

import pygame
from pygame import gfxdraw
from pygame.typing import ColorLike

import gymnasium as gym
from gymnasium import spaces

Observation = dict[str, np.typing.NDArray[Any]]
TileMap = np.typing.NDArray[np.uint8]
TileEffect = float

class Parkour(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10
    }
    TILE_SIZE = 64
    EFFECT_AMP = TILE_SIZE / 64

    def __init__(
        self, render_mode: str | None = None
    ) -> None:
        super().__init__()

        self.screen_width = self.TILE_SIZE * 16
        self.screen_height = self.TILE_SIZE * 16

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64
        )
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=np.array([0.0, 0.0, -10, -10.0, 0.0, 0.0], dtype=np.float64),
                high=np.array([self.screen_width, self.screen_height, 10, 10.0, 1.0, 1.0]),
                shape=(6,),
                dtype=np.float64
            ),
            "tile_map": spaces.Box(
                low=0,
                high=4,
                shape=(16, 16),
                dtype=np.uint8
            )
        })

        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.state: np.typing.NDArray[np.float64] | None = None
        self.steps = 0
        map_list = [ # -----------------------------------------
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1],
                [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1],
                [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.uint8),
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1],
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1],
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.uint8),
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.uint8),
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.uint8),
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 2, 2, 4, 4, 4, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
                [1, 2, 2, 4, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.uint8),
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1],
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.uint8),
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1],
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.uint8),
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 2, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=np.uint8),
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 4, 1],
                [1, 1, 1, 0, 4, 4, 1, 1, 1, 1, 4, 4, 0, 0, 0, 1],
                [1, 1, 1, 0, 4, 4, 4, 2, 2, 4, 4, 4, 0, 0, 0, 1],
                [1, 1, 1, 0, 4, 4, 4, 2, 2, 4, 4, 4, 0, 0, 0, 1],
                [1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]
            ], dtype=np.uint8)
        ] # ----------------------------------------------------
        tile_set = TileSet()
        tile_set.register(TileID.EMPTY, Tile(False, pygame.image.load(r"assets\textures\empty.png")))
        tile_set.register(TileID.WALL, Tile(True, pygame.image.load(r"assets\textures\wall.png")))
        tile_set.register(TileID.FINISH, Tile(False, pygame.image.load(r"assets\textures\finish.png")))
        # tile_set.register(TileID.EMPTY_PUNISH, Tile(False, pygame.image.load(r"assets\textures\empty_punish.png")))
        self.level = Level(tile_map=map_list, tile_set=tile_set)

    def step(self, action: np.typing.NDArray[np.float64]) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions."""
        assert self.action_space.contains(action), (
            f"Action '{action!r}' ({type(action)}) is invalid"
        )
        self.steps += 1

        assert self.state is not None
        x, y, x_velocity, y_velocity, grounded, obstructed = self.state # GET STATE
        previous_state = self.state

        hitbox = self.get_hitbox(x, y)
        grounded = int(self.box_collides( # refresh grounded state
            hitbox["left"] + 1,
            hitbox["right"] - 1,
            hitbox["bottom"] - 1,
            hitbox["bottom"]
        ))

        if x_velocity != 0 and np.sign(x_velocity) != np.sign(action[0]):
            x_velocity = 0

        x_velocity += min(10 * self.EFFECT_AMP, max(-10 * self.EFFECT_AMP, action[0] * 3 * self.EFFECT_AMP)) # Create X velocity
        if action[1] > 0: # jumping
            if grounded:
                y_velocity = min(20 * self.EFFECT_AMP, y_velocity + 1 + action[1] * 20) # Create upwards velocity allowing for jumps
        else: # diving
            if not grounded:
                y_velocity = max(-20 * self.EFFECT_AMP, y_velocity + 1 + action[1] * 5) # Create downwards velocity allowing for dives

        x += x_velocity
        hitbox = self.get_hitbox(x, y)

        if self.box_collides(hitbox["left"], hitbox["right"], hitbox["bottom"], hitbox["top"]):
            if x_velocity > 0: # moving right
                tile_x = int(hitbox["right"] // self.TILE_SIZE)
                x = tile_x * self.TILE_SIZE - self.TILE_SIZE / 4
            elif x_velocity < 0: # moving left
                tile_x = int(hitbox["left"] // self.TILE_SIZE)
                x = (tile_x + 1) * self.TILE_SIZE + self.TILE_SIZE / 4

            # y_velocity += x_velocity / 2
            x_velocity *= -0.5
            

        y += y_velocity
        hitbox = self.get_hitbox(x, y)

        if self.box_collides(hitbox["left"], hitbox["right"], hitbox["bottom"], hitbox["top"]):
            if y_velocity > 0: # going up
                tile_y = int(hitbox["top"] // self.TILE_SIZE)
                y = tile_y * self.TILE_SIZE - hitbox["height"]
            elif y_velocity < 0: # falling
                tile_y = int(hitbox["bottom"] // self.TILE_SIZE)
                y = (tile_y + 1) * self.TILE_SIZE

            y_velocity = 0

        if not grounded:
            y_velocity -= 1

    
        self.state = np.array((x, y, x_velocity, y_velocity, grounded, obstructed), dtype=np.float64) # SET STATE
        occupied_tile = self.level.get_tile(int(x), int(y), self.TILE_SIZE)
        prev_occupied_tile = self.level.get_tile(int(previous_state[0]), int(previous_state[1]), self.TILE_SIZE)
 
        reward = -0.001
        finished = occupied_tile == self.level.tile_set.get(TileID.FINISH) # If the character is on a finish tile

        occupied_effect = self.level.get_tile_effect(int(x), int(y), self.TILE_SIZE)
        previous_effect = self.level.get_tile_effect(int(previous_state[0]), int(previous_state[1]), self.TILE_SIZE)
        if occupied_effect != 0.0 and not np.isclose(previous_effect, occupied_effect):
            reward += occupied_effect
        if np.array_equal(self.state, previous_state): reward -= 0.1
        if finished: reward += 100

        terminated = bool(
            not 0 < x + x_velocity < self.screen_width or
            not 0 < y < self.screen_height or
            finished
        )
        truncated = False
        info = {
            "stepped": self.steps
        }

        if self.render_mode == "human":
            self.render()

        observation = {
            "state": self.state,
            "tile_map": self.level.tile_map
        }
        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Observation, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info."""
        super().reset(seed=seed, options=options)

        obs = np.array([
            self.TILE_SIZE * 8, # X position
            self.TILE_SIZE * 10, # Y position
            0.0, # X velocity
            0.0, # Y velocity
            1.0, # Grounded 0/1
            1.0 # Obstructed 0/1 
        ], dtype=np.float64)
        self.state = obs

        if hasattr(self.level, "map_list"):
            self.level.change_map()

        observation = {
            "state": self.state,
            "tile_map": self.level.tile_map
        }
        info: dict[str, Any] = {}
        # pygame.time.wait(140)

        return observation, info

    def render(self) -> np.typing.NDArray | None:
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255,255,255))

        # Start drawing to self.surf here (with bottom-left as origin). 
        self.level.render(self.surf, self.TILE_SIZE)

        if self.state is None:
            return None

        x = self.state
        charx = self.state[0]
        chary = self.state[1]
        vecx = self.state[2]
        vecy = self.state[3]
        next_x = charx + vecx
        next_y = chary + vecy
        
        self.level.render_character(self.surf, max(0, charx), max(0, chary), self.TILE_SIZE, text=f"({charx:.1f}, {chary:.1f}) <{vecx:.1f}, {vecy:.1f}> {bool(self.state[3])}")
        self.level.render_character(self.surf, max(0, next_x), max(0, next_y), self.TILE_SIZE, color=(0, 0, 0, 20))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            pygame.font.quit()
            pygame.quit()

    def get_hitbox(self, x: float, y: float) -> dict[str, float]:
        half_width = self.TILE_SIZE / 4
        height = self.TILE_SIZE - self.TILE_SIZE / 16

        return {
            "left": x - half_width,
            "right": x + half_width,
            "bottom": y,
            "top": y + height,
            "height": height
        }

    def box_collides(self, left: float, right: float, bottom: float, top: float) -> bool:
        tile_size = self.TILE_SIZE

        min_tx = int(left // tile_size)
        max_tx = int((right - 1) // tile_size)
        min_ty = int(bottom // tile_size)
        max_ty = int((top - 1) // tile_size)

        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                try:
                    tile = self.level.tile_set.get(int(self.level.tile_map[ty, tx]))
                    if tile.collidable:
                        return True
                except IndexError:
                    continue

        return False

class TileID(IntEnum):
    EMPTY = 0
    WALL = 1
    FINISH = 2
    EMPTY_PUNISH = 3
    EMPTY_REWARD = 4


@dataclass(frozen=True)
class Tile:
    collidable: bool
    texture: pygame.Surface
    custom_renderer: Callable | None = None


class TileSet:
    def __init__(self) -> None:
        self._tiles: dict[TileID, Tile] = {}

    def register(self, tile_id: TileID, tile: Tile) -> None:
        if tile_id in self._tiles:
            raise ValueError(f"Tile {tile_id} already exists")

        self._tiles[tile_id] = tile

    @staticmethod
    def _normalize_tile_id(tile_value: int) -> TileID:
        if tile_value in {TileID.EMPTY.value, 3, 4}:
            return TileID.EMPTY
        return TileID(tile_value)

    def get(self, tile_id: TileID | int) -> Tile:
        if isinstance(tile_id, int):
            tile_id = self._normalize_tile_id(tile_id)
        return self._tiles[tile_id]


class Level:
    """Base class for all levels"""
    def __init__(self, tile_map: TileMap | list[TileMap], tile_set: TileSet) -> None:
        self.tile_set = tile_set

        if isinstance(tile_map, list):
            if len(tile_map) > 0:
                self.map_list = tile_map
                self.tile_map = np.flipud(random.choice(tile_map))
                
            else:
                raise ValueError("Received an empty list where a non-empty list is required")

        else:
            self.tile_map: np.typing.NDArray[np.uint8] = np.flipud(tile_map)
            
        pygame.font.init()
        self.font = pygame.font.Font(r"assets\fonts\Commissioner-Regular.ttf", 20)

    def _normalize_tile_id(self, tile_value: int) -> TileID:
        if tile_value in {TileID.EMPTY.value, 3, 4}:
            return TileID.EMPTY
        return TileID(tile_value)

    def get_tile_effect(self, x: int, y: int, tile_size: int) -> TileEffect:
        """Returns the effect of the tile at a pixel coordinate."""
        tile_x, tile_y = self.pixel_to_tile(x, y, tile_size)
        try:
            tile_value = int(self.tile_map[tile_y, tile_x])
        except IndexError:
            return 0.0

        if tile_value == 3:
            return -2.0
        if tile_value == 4:
            return 2.0
        return 0.0

    def render(self, surface: pygame.Surface, tile_size: int) -> None:
        """Renders the level onto a surface, excluding player"""
        for y in range(self.tile_map.shape[0]):
            for x in range(self.tile_map.shape[1]):
                tile_id = self._normalize_tile_id(int(self.tile_map[y, x]))
                tile = self.tile_set.get(tile_id)

                screen_x = x * tile_size
                screen_y = y * tile_size

                surf = pygame.transform.flip(tile.texture, False, True)
                rect = pygame.Rect(
                    screen_x,
                    screen_y,
                    tile_size,
                    tile_size
                )

                surface.blit(surf, rect)

    def render_character(self, surface: pygame.Surface, x: int, y: int, tile_size: int, color: ColorLike = (255, 0, 0), text: str | None = None) -> None:
        """Draws the character texture on a surface"""
        x = round(x)
        y = round(y)
        left = x - tile_size / 4
        right = x + tile_size / 4
        bottom = y
        top = y + tile_size - tile_size / 16
        gfxdraw.aapolygon(surface, [(left, bottom), (right, bottom), (right, top), (left, top)], color) # Temp texture, renders the hitbox
        if text is not None:
            text_surf = pygame.transform.flip(self.font.render(text, True, (0,0,0)), False, True)
            text_rect = text_surf.get_rect(midtop=(x, top))
            surface.blit(text_surf, text_rect)

    def change_map(self, tile_map: TileMap | None = None) -> None:
        """Updates the tile map. None is only valid when the level maintains a list of tiles. This does not update character coordinates"""
        if hasattr(self, "map_list"):

            if tile_map is not None:
                raise ValueError("Received a TileMap instance, but None was expected.")
            self.tile_map = np.flipud(random.choice(self.map_list))

        else:
            if tile_map is not None:
                self.tile_map = np.flipud(tile_map)

    def pixel_to_tile(self, x: int, y: int, tile_size: int) -> tuple[int, int]:
        """Converts pixel coordinates to tile grid coordinates. Returns (x, y)."""
        tile_x = x // tile_size
        tile_y = y // tile_size
    
        return tile_x, tile_y

    def check_collision(self, x: int, y: int, tile_size: int) -> bool:
        """Checks if a point on screen is colliding with a tile with collision enabled"""
        tile_x, tile_y = self.pixel_to_tile(x, y, tile_size)
        try:
            tile = self.tile_set.get(self._normalize_tile_id(int(self.tile_map[tile_y, tile_x])))
        except IndexError:
            return False
        return getattr(tile, "collidable") # Checks if the tile has collision and returns
 
    def get_tile(self, x: int, y: int, tile_size: int) -> Tile:
        """Returns the tile at a pixel coordinate"""
        tile_x, tile_y = self.pixel_to_tile(x, y, tile_size)
        try:
            tile = self.tile_set.get(self._normalize_tile_id(int(self.tile_map[tile_y, tile_x])))
        except IndexError:
            return Tile(False, pygame.Surface((64,64), masks=(255,255,255)))
        return tile