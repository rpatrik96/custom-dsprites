from os import makedirs
from os.path import dirname, abspath, join, isdir

import numpy as np
import torch

from args import parse_args
from spriteworld.custom_generators import generate_isprites
from .transforms import (
    projective_transform,
    affine_transform,
    hsv_change,
)
from .utils import sprites_filename, to_one_hot


def sprites_gen_wrapper(nobs, nclasses, projective, affine, deltah, deltas, deltav, angle, shape, lower, upper):
    # init
    nfactors = 4
    beta_params = (
        torch.Tensor(np.random.uniform(lower, upper, 2 * nfactors * nclasses)).view(nclasses, nfactors, 2).numpy())
    angle_params = torch.zeros((nclasses, 2)).numpy()
    shape_probs = torch.zeros((nclasses, 3)).numpy()

    if angle is True:
        nfactors += 1
    if shape is True:
        nfactors += 1

    obs_per_class = int(nobs / nclasses)

    # generate
    S = np.zeros((nclasses, obs_per_class, nfactors))
    X, Y = generate_isprites(nclasses, obs_per_class, beta_params, S, angle, angle_params, shape_probs, shape, lower,
                             upper)
    S = torch.Tensor(S).flatten(0, 1).numpy().astype(np.float32)
    Y = to_one_hot(Y)[0].astype(np.float32)

    # add disturbance
    if projective is True:
        X = projective_transform(X)
    if affine is True:
        X = affine_transform(X)
    if deltah != 0 or deltas != 0 or deltav != 0:
        print("Applying color transformation in HSV space...")
        X = np.array([hsv_change(x, deltah, deltas, deltav) for x in X])

    # save
    sprites_dir = join(dirname(dirname(abspath(__file__))), "sprites_data")
    if not isdir(sprites_dir):
        makedirs(sprites_dir)

    filename = sprites_filename(nobs, nclasses, projective, affine, deltah != 0 or deltas != 0 or deltav != 0, shape,
                                angle, lower, upper, extension=False )

    np.savez_compressed(
        join(sprites_dir, filename), X, Y, S, beta_params, angle_params, shape_probs
    )

    return X, Y, S


if __name__ == "__main__":
    # Command line arguments
    args = parse_args()

    sprites_gen_wrapper(args.nobs, args.nclasses, args.projective, args.affine, args.deltah, args.deltas, args.deltav,
                        args.angle, args.shape, args.lower, args.upper)
