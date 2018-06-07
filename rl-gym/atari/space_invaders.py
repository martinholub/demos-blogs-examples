import gym
import gym.spaces
from gym.wrappers import Monitor
gym.logger.set_level(40) # Disable printing of warn level log to stdout

#import cv2
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import matplotlib
matplotlib.use("Agg") # use noninteractive backend
import matplotlib.pyplot as plt
import subprocess

from duel_Q import DuelQ
from deep_Q import DeepQ

from prioritized_memory import PERMemory as Memory
from decorate import func_args

from logger_utils import initialize_logger, teardown_logger
logger = initialize_logger() # logging is handled by job manager

class SpaceInvader(object):
    # TODO: some checks and assertions

    def __init__(self, env_id = "SpaceInvaders-v0", mode = "DQN",
                replay_size = 64, num_kept_frames = 4, max_memory = 20000,
                discount = 0.99, epsilon = 1., eps_decay_rate = 0.995,
                eps_min = 0.1, max_steps = 50000, min_steps_train = 10000,
                num_episodes = 1000, img_size = (84, 84), learn_rate = 0.01,
                max_frames = 1500000, target_update_freq = 5000,
                verbose = True, load_path = None):
        # Enviroment
        self._make_env(env_id)
        self.verbose = bool(verbose)
        # Model
        self.img_size = tuple(img_size)
        self.discount = float(discount) # discount rate (also gamma)
        self.epsilon = float(epsilon) # exploration rate
        self.eps_decay_rate = float(eps_decay_rate) # ncrease in model certainity
        self.eps_min = float(eps_min) # minimal exploration rate
        self.max_steps = int(max_steps) # number of total frames per episode
        self.min_steps_train = int(min_steps_train) # Step no when training starts
        self.num_episodes = int(num_episodes) # Number of games to play
        self.max_frames = int(max_frames) # number of total frames
        self.learn_rate = float(learn_rate)
        self.replay_size = int(replay_size)
        self.target_update_freq = int(target_update_freq)
        # Memory
        self.memory = Memory(max_memory)
        self.num_kept_frames = int(num_kept_frames)
        self.process_buffer = self._init_process_buffer()

        # Construct appropriate network based on flags
        self.mode = mode
        q_args=(self.learn_rate, self.img_size, self.num_kept_frames,
                self.action_size, self.replay_size)
        if mode == "DDQN":
            print("DDQN model is implemented but not tested, thus not encouraged.")
            self.agent = DeepQ(*q_args)
        elif mode == "DQN":
            self.agent = DuelQ(*q_args)

        if load_path is not None:
            self.load_network(load_path)

    def _make_env(self, name):
        self.env = gym.make(name)
        _ = self.env.reset()
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        logger.info(  "There are {} actions in the enviroment: {}".\
                format(self.action_size, self.env.unwrapped.get_action_meanings()))


    def _init_process_buffer(self):
        process_buffer = []
        # A buffer that keeps the last 3 images
        for i in range(self.num_kept_frames):
            # Initialize buffer with the first frame
            state, _, _, _ = self.env.step(0)
            process_buffer.append(state)
        return(process_buffer)

    def load_network(self, path):
        self.agent.load_network(path)

    def convert_process_buffer(self):
        """Converts the list of num_kept_frames images in the process buffer
        into one training sample"""
        imsize = self.img_size
        gray_buffer = [ resize(rgb2gray(x), imsize, mode = "constant") \
                        for x in self.process_buffer]
        gray_buffer = [np.reshape(x, x.shape + (1, )) for x in gray_buffer]

        return(np.concatenate(gray_buffer, axis=2).astype(np.float32))

    def remember(self, state, action, reward, next_state, done):
        """Prevent NN forgeting about past experiences
        """
        self.memory.append((state, action, reward, next_state, done))

    def remember_priority(self, state, action, reward, next_state, done):
        """ Remeber also error at current single step

        This allows us to sample from memory with priority.
        Of course, it also slows us down a bit. """

        _, _, error=self.agent.predict_sample(  state, action, reward, next_state,
                                                done, self.discount)
        # Assign highest error to recent experience
        # error = [1e6]
        self.memory.add(error[0], (state, action, reward, next_state, done))

    def train(self):
        self.episode_rewards = []
        frame_no = 0
        # initial state == current state if ep == 0
        initial_state = self.convert_process_buffer()

        for ep in range(self.num_episodes):
            # Reset the environment
            _ = self.env.reset()

            state = initial_state # Starting a clean game
            episode_reward = 0
            done = False
            for t in range(self.max_steps):
                self.process_buffer = [] # clean up buffer
                action, q_value = self.agent.policy(state, self.epsilon)
                # FRAME SKIPPING: Fill the buffer
                for i in range(self.num_kept_frames):
                    state_frame, reward, done_frame, _ = self.env.step(action)
                    reward = np.clip(reward, -1., 1.)
                    episode_reward += reward
                    done = done | done_frame
                    self.process_buffer.append(state_frame)
                    frame_no +=1

                next_state = self.convert_process_buffer()
                self.remember_priority(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state

                if (t%10 == 9) and self.verbose:
                    logger.debug("Action: {}, Q_val: {}".format(action, q_value))

                if (frame_no%self.target_update_freq>=(self.target_update_freq-self.num_kept_frames)):
                    self.agent.target_update()
                    logger.info("Target model updated, frame: {}".format(frame_no))

                if done: # each episode is allowed to take max_steps or until we die
                    break

            # PRIORITIZED EXPERIENCE REPLAY: Train with once per game frequency
            if self.memory.tree.n_entries >= self.min_steps_train:
                # TODO: check pass by value/reference
                self.memory = self.agent.replay(self.memory, self.discount)
                logger.info("Model trained, ep: {}".format(ep+1))

            # EXPLORATION/EXPLOTATION POLICY: Increase model certainity
            if self.epsilon > self.eps_min:
                self.epsilon *= self.eps_decay_rate

            self.episode_rewards.append(episode_reward)

            if self.verbose:
                if (((ep % 15) == 0) and ep > 0):
                    mean_reward = np.mean(self.episode_rewards[max(ep-15, 0):ep+1])
                    logger.info("Episode {}/{} finished. Mean reward over last 15 episodes: {:.2f}"\
                        .format(ep, self.num_episodes, mean_reward))
                else:
                    logger.info("Episode {}/{} finished. Reward: {:.2f}, Frame # {}."\
                        .format(ep+1, self.num_episodes, episode_reward, frame_no))


            # Save the network 250 ep
            if ((ep % 250) == 0 and ep > 0):
                logger.info("Saving Network, frame # {}".format(frame_no))
                self.agent.save_network("saved_models/" + \
                                        self.mode+"_test_"+str(frame_no)+".h5")

            if frame_no >= self.max_frames or ep == (self.num_episodes - 1) :
                logger.info("Frame # {}. End of Training.".format(frame_no))
                break

    def plot_(self):
        """Simple plot wrapper
        """
        try:
            plt.clf()
            fig, ax  = plt.subplots(1, 1)
            ax.plot(self.episode_rewards)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            # ax.axhline(self.ep.reward_solve, 0, 1, alpha = 0.5, ls = "--", lw = "1.0",
                    # color = "firebrick")
            fig.savefig("summary/EpisodeRewards_spaceinvaders.png")
            logger.info("Plot was saved.")
        except NameError as e: # or whatever gets thrown if not defined
            raise e
        pass

    def simulate_(self):
        """Simple game visualizer"""
        done = False
        tot_award = 0
        state = self.env.reset()
        frame_no = 0
        while not done:
            frame_no += 1
            state = self.convert_process_buffer()
            action = self.agent.policy(state, 0)[0]
            state, reward, done, _ = self.env.step(action)
            tot_award += reward

            # TODO: May be more elegant to pop
            self.process_buffer.append(state)
            self.process_buffer = self.process_buffer[1:]

            frame = self.env.render(mode = 'rgb_array')
            plt.imsave(*("anim/{}.png".format(frame_no), frame),
                        **{"vmin":0, "vmax":255})

        self.env.reset()
        self.env.close()

        print("Lived for {} frames, total reward was {}".format(frame_no, tot_award),
                flush = True)

        subprocess.run("ffmpeg -y -f image2 -i anim/%d.png anim/video.avi",
                        shell = True)
        subprocess.run("ffmpeg -y -i anim/video.avi -r 9 anim/anim.gif",
                        shell = True)


    def main(self):
        logger = initialize_logger()
        try:
            self.train()
        finally:
            teardown_logger(logger)


    # def calculate_mean(self, num_samples = 100):
    #     reward_list = []
    #     print("Printing scores of each trial")
    #     for i in range(num_samples):
    #         done = False
    #         tot_award = 0
    #         self.env.reset()
    #         while not done:
    #             state = self.convert_process_buffer()
    #             action = self.agent.policy(state, 0.0)[0]
    #             state, reward, done, _ = self.env.step(action)
    #             tot_award += reward
    #             self.process_buffer.append(state)
    #             self.process_buffer = self.process_buffer[1:]
    #
    #         print(tot_award)
    #         reward_list.append(tot_award)
    #     return(np.mean(reward_list), np.std(reward_list))
