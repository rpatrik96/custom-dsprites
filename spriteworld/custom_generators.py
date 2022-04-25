import numpy as np


import environment as spriteworld_environment
from config import random_sprites_config


def collect_frames(config, label, num_frames, angle:bool, shape:bool, S):
    """Instantiate config as environment and get single images from it.
    :param angle:
    :param shape:
    """
    env = spriteworld_environment.Environment(**config)
    images = []
    for i in range(num_frames):
        ts = env.reset()
        S[label, i, 0] = env._sprites[0].x[0]
        S[label, i, 1] = env._sprites[0].y[0]
        S[label, i, 2] = env._sprites[0].scale[0]
        S[label, i, 3] = env._sprites[0].c0[0]
        if angle is True:
            S[label, i, 4] = env._sprites[0].angle[0]
        if shape is True:
            if env._sprites[0].shape == "triangle":
                S[label, i, 5] = 0
            elif env._sprites[0].shape == "square":
                S[label, i, 5] = 1
            elif env._sprites[0].shape == "pentagon":
                S[label, i, 5] = 2

        images.append(ts.observation["image"])
    return images


def generate_isprites(num_classes, obs_per_class, beta_params, S, angle, angle_params, shape_probs, shape, lower,
                      upper):
    for i in range(num_classes):
        print(i)
        if i == 0:
            full_obs = collect_frames(
                random_sprites_config(beta_params, i, angle, angle_params, shape_probs, shape, lower, upper), i,
                obs_per_class, angle, shape, S)
            full_labels = np.zeros(obs_per_class)
        else:
            full_obs += collect_frames(
                random_sprites_config(beta_params, i, angle, angle_params, shape_probs, shape, lower, upper), i,
                obs_per_class, angle, shape, S)
            full_labels = np.concatenate((full_labels, np.ones(obs_per_class) * i))

    return np.array(full_obs), np.array(full_labels)
