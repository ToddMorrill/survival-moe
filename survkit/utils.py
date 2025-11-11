import math
import os
import pickle
import random
import numpy as np
import torch

from .configs import ArgParser, WandbConfig, TrainConfig

def get_args(command_string=None):
    """Accepts an optional command string, which will be parsed into the config classes. Otherwise, the command line arguments will be parsed."""
    args = None
    if command_string is not None:
        args = command_string.split()
    parser = ArgParser([WandbConfig, TrainConfig])
    wandb_config, train_config, unknown_args = parser.parse_args_into_dataclasses(args=args,
        return_remaining_strings=True,
        args_file_flag='--experiment_args',)
    if unknown_args:
        raise ValueError(f"Unknown args: {unknown_args}")
    return wandb_config, train_config

def get_colors(palette='tableau'):
    if palette == 'jax':
        light_pink = np.array([224, 135, 248])/255
        light_blue = np.array([104, 150, 242])/255
        light_green = np.array([73, 163, 154])/255

        pastel_pink = np.array([208, 145, 216])/255
        pastel_blue = np.array([173, 201, 252])/255
        pastel_green = np.array([159, 210, 205])/255

        dark_pink = np.array([100, 35, 151])/255
        dark_blue = np.array([50, 84, 194])/255
        dark_green = np.array([37, 103, 93])/255

        colors = dict(pastel_pink=pastel_pink, light_pink=light_pink, dark_pink=dark_pink,
                        pastel_blue=pastel_blue, light_blue=light_blue, dark_blue=dark_blue,
                        pastel_green=pastel_green, light_green=light_green, dark_green=dark_green)
    elif palette == 'tableau':
        # get the tableau colors
        neuron_colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
        neuron_colors = [(r/255, g/255, b/255) for r, g, b in neuron_colors]
        color_names = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'yellow', 'cyan']
        color_shades = ['dark', 'light']
        colors = [f'{shade}_{name}' for name in color_names for shade in color_shades]
        colors = dict(zip(colors, neuron_colors))
    return colors

def get_cosine_annealing_multiplier(current_step, hold_steps, anneal_steps):
    """Calculates a multiplier using a cosine schedule based on the current step."""
    # calculate how far into the annealing phase we are
    anneal_progress = (current_step - hold_steps) / (anneal_steps - 1)

    # cosine annealing formula: 0.5 * (1 + cos(pi * progress))
    # this smoothly transitions from 1 to 0
    return 0.5 * (1 + math.cos(math.pi * anneal_progress))

def get_lambda_multiplier(current_step, hold_steps, anneal_steps, min_val=0.0):
    """Calculates a lambda multiplier based on the current step, hold steps, and anneal steps."""
    lambda_multiplier = min_val
    if current_step < hold_steps:
        # multiplier is 1.0 during hold phase
        lambda_multiplier = 1.0
    elif current_step < hold_steps + anneal_steps:
        # calculate multiplier using the step-based function during anneal phase
        lambda_multiplier = get_cosine_annealing_multiplier(current_step, hold_steps, anneal_steps)
        # alternatively, you could use exponential annealing
        # lambda_multiplier = get_exponential_annealing_multiplier(
        #     current_step, hold_steps, anneal_steps)
    # respect min_val
    lambda_multiplier = max(lambda_multiplier, min_val)
    return lambda_multiplier