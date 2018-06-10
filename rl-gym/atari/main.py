#!/usr/bin/env python
import argparse
from space_invaders import SpaceInvader


ap = argparse.ArgumentParser(description="Train and test different networks on Space Invaders")

# Parse arguments
ap.add_argument("-n", "--network", type=str, action='store',
                help="Please specify the network you wish to use, either DQN or DDQN",
                required=True, choices = ["DQN", "DDQN"])
ap.add_argument("-m", "--mode", type=str, action='store',
                help="Please specify the mode you wish to run, either train or test",
                required=True, choices = ["train", "test"])
ap.add_argument("-l", "--load", type=str, action='store',
                help="Please specify the file you wish to load weights from(for example saved.h5)",
                required=False)
ap.add_argument("-s", "--save", action='store_true', required=False,
                help="Save animation of simulation")

args = ap.parse_args()
print(args)

game = SpaceInvader(mode = args.network)

if args.load:
    game.load_network(args.load)

if args.mode == "train":
    try:
        game.main()
    finally:
        num_ep = len(game.episode_rewards)
        game.agent.save_network("saved_models/spaceinvader_ep{}.h5".format(num_ep))

if args.save:
    try:
        game.simulate_()
    except Exception as e:
        print("Simulation did not work!")
        raise e
