#!/usr/bin/env python
import argparse
from atari_game import AtariGame


ap = argparse.ArgumentParser(description="Train and test different networks on Space Invaders")

# Parse arguments
ap.add_argument("-n", "--network", type=str, action='store',
                help="Please specify the network you wish to use, either DQN or DDQN",
                required=False, choices = ["DQN", "DDQN"], default = "DQN")
ap.add_argument("-m", "--mode", type=str, action='store',
                help="Please specify the mode you wish to run, either train or test",
                required=False, choices = ["train", "test"], default = "train")
ap.add_argument("-e", "--env", type=str, action='store', default = "BreakoutDeterministic-v4",
                help="Please specify the environment id.", required=True)
ap.add_argument("-l", "--load", type=str, action='store',
                help="Please specify the file you wish to load weights from(for example saved.h5)",
                required=False)
ap.add_argument("-s", "--save", action='store_true', required=False,
                help="Save animation of simulation")

args = ap.parse_args()
print(args)

game = AtariGame(mode = args.network, env_id = args.env)

if args.load:
    game.load_network(args.load)

if args.mode == "train":
    try:
        game.main()
    finally:
        game.plot_()
        num_ep = len(game.episode_rewards)
        game.agent.save_network("saved_models/{}_ep{}.h5".format(args.env, num_ep))

        if args.save:
            try:
                game.simulate_()
            except Exception as e:
                print("Simulation did not work!")

elif args.mode == "test":
    game.agent.test(n_episodes = 10, visualize = False)

if args_mode != "train" & args.save:
    try:
        game.simulate_()
    except Exception as e:
        print("Simulation did not work!")
        raise e
