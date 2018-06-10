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
import random

from duel_Q import DuelQ
from deep_Q import DeepQ

from prioritized_memory import PERMemory as Memory
from collections import deque
from decorate import func_args

from logger_utils import initialize_logger, teardown_logger
logger = initialize_logger() # logging is handled also by job manager

class SpaceInvader(object):
    def __init__(self, env_id = "SpaceInvadersNoFrameskip-v4", mode = "DQN",
                replay_size = 64, num_kept_frames = 3, max_memory = 100000,
                discount = 0.9, epsilon = 1., eps_decay_rate = 0.995,
                eps_min = 0.01, max_steps = 100000, min_steps_train = 20000,
                num_episodes = 15000, img_size = (84, 84), learn_rate = 0.001,
                max_frames = 20000000, target_update_freq = 10000, model_train_freq = 200,
                verbose = True, load_path = None, model_save_freq = 1000,
                qa_report_freq = 50):

        self._make_env(env_id)
        self.verbose = bool(verbose)
        # Model
        self.img_size = tuple(img_size) # downsampling image size
        self.discount = float(discount) # discount rate (also gamma)
        self.epsilon = float(epsilon) # exploration rate
        self.eps_decay_rate = float(eps_decay_rate) # increase in model certainity
        self.eps_min = float(eps_min) # minimal exploration rate
        self.max_steps = int(max_steps) # number of total frames per episode
        self.min_steps_train = int(min_steps_train) # Step No. when training starts
        self.num_episodes = int(num_episodes) # Number of games to play
        self.max_frames = int(max_frames) # number of total frames
        self.learn_rate = float(learn_rate) # optimizer learning rate
        self.model_save_freq = int(model_save_freq) # save model every n steps
        self.model_train_freq = int(model_train_freq) # train model every n steps
        self.qa_report_freq = int(qa_report_freq) # print Q, a, eps every n steps
        self.num_kept_frames = int(num_kept_frames) # Deterministic frameskip
        self.process_buffer = self._init_process_buffer()
        # Target
        # bring target up to speed with model every n steps
        self.target_update_freq = int(target_update_freq)

        # Construct appropriate network based on flags
        self.mode = mode
        q_args=(self.learn_rate, self.img_size, self.num_kept_frames,
                self.action_size, replay_size, max_memory)
        if mode == "DDQN":
            print("DDQN model is implemented but not tested, thus not encouraged.")
            self.agent = DeepQ(*q_args)
        elif mode == "DQN":
            self.agent = DuelQ(*q_args)

        if load_path is not None:
            self.load_network(load_path)

    def _make_env(self, name):
        """Instantiate an OpenAI ennvironment

        # Beware of where frames-skipping/action-repetition happens
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        # also for limits on _max_env_steps
        # For simplicity, set num_kept_frames=3 and use '{}NoFrameskip-v4' or '{}v4'
        """
        self.env = gym.make(name)
        _ = self.env.reset()
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        # self.env._max_env_steps = self.max_steps # not needed, handled by for loop
        logger.info("Environment: {}. There are {} actions: {}".\
                format(self.env.spec.id, self.action_size,
                        self.env.unwrapped.get_action_meanings()))

    def _init_process_buffer(self):
        process_buffer = []
        # A buffer that keeps the last <num_kept_frames> images
        for _ in range(self.num_kept_frames):
            # Initialize buffer with the first frame
            state, _, _, _ = self.env.step(0)
            process_buffer.append(state)
        return(process_buffer)

    def load_network(self, path):
        self.agent.load_network(path)

    def convert_process_buffer(self):
        """Converts the list of num_kept_frames images in the process buffer
        into one training sample.

        Note that the state should be fed to NN on 0..1 scale. Memorywise, it would be
        more efficient to store the arrays as uint8 (0..255) and preprocess
        only the minibatch/single sample at model input.

        https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
        """
        imsize = self.img_size
        gray_buffer = [ resize(rgb2gray(x), imsize, mode = "constant") \
                        for x in self.process_buffer]
        gray_buffer = [np.reshape(x, x.shape + (1, )) for x in gray_buffer]

        return(np.concatenate(gray_buffer, axis=2).astype(np.float32))

    def remember(self, state, action, reward, next_state, done):
        """Prevent NN forgeting about past experiences
        """
        if isinstance(self.agent.memory, (Memory, )):
            self.remember_priority(state, action, reward, next_state, done)
        else:
            self.agent.memory.append((state, action, reward, next_state, done))

    def remember_priority(self, state, action, reward, next_state, done):
        """ Remeber also error at current single step

        This allows us to sample from memory with priority.
        Of course, it also slows us down a bit.

        https://github.com/rlcode/per
        """

        _, _, error=self.agent.predict_sample(  state, action, reward, next_state,
                                                done, self.discount)
        self.agent.memory.add(error[0], (state, action, reward, next_state, done))

    def train(self):
        """ Train the agent to drive up the total reward per episode

        Loop over `num_epsiodes`, each with `max_steps` upmost. Select the
        optimal action according to the current `policy` (given by NN weights).
        Deterministically repeat action over `num_kept_frames`, preprocess
        observations into `state`. Store (s,a,r,s',d) in memory on each step.
        Train model at `model_train_freq` frequency, each time on a `replay_size`
        samples ("minibatch") drawn from memory. Periodically, with frequency
        `target_update_freq` bring the target up to date with model. Also, periodcially
        print Q, action and eps values and save the model weights.


        Notes:
          - There are `num_kept_frames` times more frames then step.
          - If `done` inside frameskip, impute missing as all zeros.
          - Should train more often, ideally each step (gets expensive).

        https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
        http://cs229.stanford.edu/notes/cs229-notes12.pdf
        https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
        """
        self.episode_rewards = []   # Store reward achieved at each epsiode
        self.all_rewards = []       # Store reward obtained at each frame
        frame_no = 0                # The smallest unit of training
        initial_state = self.convert_process_buffer()

        for ep in range(self.num_episodes):
            # Starting a clean game.
            state_ref = self.env.reset() # Reset the environment, get reference state

            # Do we always start from the same spot?
            # If so, neednt to reinitialize buffer
            # self.process_buffer = self._init_process_buffer()#`num_kept_frames` reps of nonaction
            # initial_state = self.convert_process_buffer() # Preprocess content of buffer
            state = initial_state        # initial state == current state at each ep start

            episode_reward = 0           # This one should be increasing with more training
            done = False                 # i.e. we were killed and the game has ended
            num_lives = 3                # How many lives do we start with?

            for _ in range(self.max_steps): # self.env._max_episode_steps = self.max_steps
                self.process_buffer = [] # clean up buffer
                # Pick optimal action givent current weights in the predictiove model
                action, q_value = self.agent.policy(state, self.epsilon)

                reward = 0      # Reward for `rep_frames` repeats of action
                was_killed = False  # Was the agent killed in in this `rep_frames` unit?

                # DETERMINISTIC FRAME SKIPPING: Fill the buffer
                # "We use k = 4 for all games except Space Invaders where we noticed
                # that using k = 4 makes the lasers invisible because of the period
                # at which they blink."
                # May want to have stochastic frame skip
                # rep_frames = random.choice(range(2, self.num_kept_frames + 1))
                rep_frames = self.num_kept_frames
                for _ in range(rep_frames): # repeat action over `rep_frames`
                    state_frame, reward_frame, done_frame, env_info = self.env.step(action)
                    # In original paper, the reward was clipped to work with different games
                    # It is a strong limitation (no way to learn high-reward actions).
                    # Hence we drop it.
                    # reward_frame = np.clip(reward_frame, -1., 1.)
                    done = done | done_frame        # Are we done? (0 lives)
                    self.process_buffer.append(state_frame)
                    self.all_rewards.append(reward_frame)
                    frame_no +=1
                    reward += reward_frame
                    if done:
                        break           # dont spill over to the next episode

                for _ in range(self.num_kept_frames - len(self.process_buffer)):
                    # Pad the array with empty observations, if necessary
                    self.process_buffer.append(np.zeros(state_ref.shape, np.float32))

                if env_info["ale.lives"] != num_lives:
                    num_lives -= 1
                    was_killed = True
                # Penalize terminal states. Penalize also static actions (??)

                reward += (was_killed * -10.0) #+ (reward == 0) * -1.0
                episode_reward += reward
                next_state = self.convert_process_buffer() # Preprocess content of buffer
                # Store (s,a,r,s',d) in memory
                self.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state

                state_no = frame_no // self.num_kept_frames
                # PRIORITIZED EXPERIENCE REPLAY: Train with given freq
                if (state_no % self.model_train_freq == (self.model_train_freq-1)):
                    if isinstance(self.agent.memory, (Memory, )):     # Prioritized
                        in_memory = self.agent.memory.tree.n_entries  # SumTree
                    else:
                        in_memory = len(self.agent.memory)            # deque
                    if in_memory >= self.min_steps_train:
                        # TODO: how to pass Logger instance between modules?
                        logger.info("Model trained, frame: {}".format(frame_no))
                        self.agent.replay(self.discount, logger)


                # TARGET DQN: Update target NN less frequently than model DQN
                if (state_no%self.target_update_freq == (self.target_update_freq-1)):
                    self.agent.target_update()
                    logger.info("Target updated, frame: {}".format(frame_no))

                if self.verbose: # Report some values for debugging
                    if (state_no%self.qa_report_freq == (self.qa_report_freq - 1)):
                        logger.debug("Action: {}, Q_val: {},  eps: {}".\
                                    format(action, q_value, self.epsilon))

                if done: # each episode is allowed to take max_steps or until we die
                    break

            # EXPLORATION/EXPLOTATION POLICY: Increase model certainity
            if self.epsilon > self.eps_min:
                self.epsilon *= self.eps_decay_rate

            self.episode_rewards.append(episode_reward)

            if self.verbose: # Report some values for debugging
                if (((ep % 15) == 0) and ep > 0):
                    mean_reward = np.mean(self.episode_rewards[max(ep-15, 0):ep+1])
                    logger.info("Episode {}/{} finished. Mean reward over last 15 episodes: {:.2f}"\
                        .format(ep, self.num_episodes, mean_reward))
                else:
                    logger.info("Episode {}/{} finished. Reward: {:.2f}, Frame # {}."\
                        .format(ep+1, self.num_episodes, episode_reward, frame_no))


            if ((ep % self.model_save_freq) == 0 and ep > 0):
                logger.info("Saving Network, frame: {}, ep: {}".format(frame_no, ep+1))
                self.agent.save_network("saved_models/" + \
                                        self.mode+"_test_"+str(frame_no)+".h5")

            if frame_no >= self.max_frames or ep == (self.num_episodes - 1) :
                logger.info("Frame # {}. End of Training.".format(frame_no))
                logger.info("Saving Network, frame: {}, ep: {}".format(frame_no, ep+1))
                self.agent.save_network("saved_models/" + \
                                        self.mode+"_test_"+str(ep)+".h5")
                break

    def plot_(self):
        """Plot all epsiode rewards collected throughout training
        """
        try:
            plt.clf()
            fig, ax  = plt.subplots(1, 1)
            ax.plot(self.episode_rewards)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            # ax.axhline(self.ep.reward_solve, 0, 1, alpha = 0.5, ls = "--", lw = "1.0",
                    # color = "firebrick")
            fig.savefig("summary/" + self.env.spec.id + "_Erewards.png")
            logger.info("Plot was saved.")
        except NameError as e: # or whatever gets thrown if not defined
            raise e
        pass

    def simulate_(self):
        """Simple game visualizer

        Usses ffmpeg to convert simulated images into animated gif. Make sure
        it is installed on your machine.
        """
        done = False
        tot_award = 0
        state = self.env.reset()
        frame_no = 0
        while not done:
            frame_no += 1
            state = self.convert_process_buffer()
            # Take deterministic action (eps=0) according to model.
            action = self.agent.policy(state, epsilon = 0)[0]
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
        """ Training wrapper
        """
        logger = initialize_logger()
        try:
            self.train()
        finally:
            self.plot_()
            # game.simulate_()
            teardown_logger(logger)
