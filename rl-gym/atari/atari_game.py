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
import os
import shutil

from duel_Q import DuelQ
from deep_Q import DeepQ

from prioritized_memory import PERMemory as Memory
from collections import deque
from decorate import func_args

from logger_utils import initialize_logger, teardown_logger
logger = initialize_logger() # logging is handled also by job manager

from callbacks import (UpdateLossMask, FileLogger, ModelIntervalCheckpoint, CallbackList,
                        TrainIntervalLogger, TrainEpisodeLogger, SubTensorBoard)
from keras.callbacks import History, Callback
from utils import clipped_masked_error

class AtariGame(object):
    def __init__(self, mode = "DQN", replay_size = 40, min_steps_train = 20000,
                #env_id = "SpaceInvadersNoFrameskip-v4", num_kept_frames = 3,
                env_id = "BreakoutDeterministic-v4", num_kept_frames = 4,
                discount = 0.99, epsilon = 1., eps_decay_rate = None,
                eps_min = 0.1, max_ep_steps = None, max_memory = None,
                num_episodes = 500, img_size = (80, 80), learn_rate = 0.01,
                max_frames = 50000000, target_update_freq = 10000, model_train_freq = 1,
                verbose = True, load_path = None, model_save_freq = 1000,
                qa_report_freq = 50, is_test = False):

        self._make_env(env_id)
        self.verbose = bool(verbose)
        self.is_test = bool(is_test)
        self.is_init = bool(False)
        # Model
        self.img_size = tuple(img_size) # downsampling image size
        self.discount = float(discount) # discount rate (also gamma)
        self.epsilon = np.float32(epsilon) # exploration rate
        self.eps_max = np.float32(epsilon)
        self.eps_min = float(eps_min) # minimal exploration rate
        self.min_steps_train = int(min_steps_train) # Step No. when training starts
        self.num_episodes = int(num_episodes) # Number of games to play
        self.max_frames = int(max_frames) # number of total frames
        self.learn_rate = float(learn_rate) # optimizer learning rate

        if not max_ep_steps:
            self.max_ep_steps = int(self.env._max_episode_steps)
        else:
            self.max_ep_steps = int(max_ep_steps) # number of max frames per episode

        if not model_save_freq: # could do stuff like this via @property and setter
            model_save_freq = self.num_episodes // 10
        self.model_save_freq = int(model_save_freq) # save model every n episode

        try:
            self.model_train_freq = int(model_train_freq) # train model every n steps
        except TypeError as e:
            self.model_train_freq = int(replay_size)

        if not qa_report_freq:
            qa_report_freq = self.model_train_freq
        self.qa_report_freq = int(qa_report_freq) # print Q, a, eps every n steps

        self.num_kept_frames = int(num_kept_frames) # Deterministic frameskip
        self.process_buffer = self._init_process_buffer()
        # Target
        # bring target up to speed with model every n steps
        if not target_update_freq:
            target_update_freq = min(1000*self.model_train_freq, 5000)
        self.target_update_freq = int(target_update_freq)

        if not max_memory:
            max_memory = int(self.min_steps_train*10)

        if not eps_decay_rate:
            # Go to eps_min over third of training, exponentially
            self.eps_decay_rate = np.float32(np.power((self.eps_min / self.epsilon),
                                                (1 / (self.num_episodes/3))))
        else:
            self.eps_decay_rate = np.float32(eps_decay_rate) # increase in model certainity

        # Construct appropriate network based on flags
        self.mode = mode
        q_args=(self.learn_rate, self.img_size, self.num_kept_frames,
                self.action_size, replay_size, max_memory, self.is_test,
                self.num_episodes)


        if mode == "DDQN":
            raise NotImplementedError()
        elif mode == "DQN":
            self.agent = DuelQ(*q_args)

        if load_path is not None:
            self.load_network(load_path)



    def _make_env(self, name):
        """Instantiate an OpenAI ennvironment

        Beware of where frames-skipping/action-repetition happens
        https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        also for limits on _max_env_steps
        For simplicity, set num_kept_frames=3 and use '{}NoFrameskip-v4' or '{}-v4'
        For faster training, can use '{}Deterministic-v4' which skips frames
        Can check in game.env.unwrapped.frameskip
        """
        self.env = gym.make(name)
        _ = self.env.reset()
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        # self.env._max_env_steps = self.max_ep_steps # not needed, handled by for loop
        logger.info("Environment: {}. There are {} actions: {}. Frameskip: {}".\
                format(self.env.spec.id, self.action_size,
                        self.env.unwrapped.get_action_meanings(),
                        self.env.unwrapped.frameskip))

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

    def get_curr_eps(self, ep):
        """ Anneal epsilon value linearly between max and min over third of episodes
        """
        eps_diff = self.eps_max - self.eps_min
        return (self.eps_max) - (ep/self.num_episodes*3)*eps_diff

    def to_grayscale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)

    def crop_to_boundbox(self, img, box = (34,0,194,160)):
        """Crop input 210,160,3 to 160,160,3"""
        # return img[34:194,0:160, :]
        return img[box[0]:box[2],box[1]:box[3], :]

    def downsample(self, img):
        return img[::2,::2]

    def preprocess(self, img):
        img_out = self.to_grayscale(self.downsample(self.crop_to_boundbox(img)))
        assert img_out.shape[:2] == self.img_size
        return img_out

    def convert_process_buffer(self):
        """Converts the list of num_kept_frames images in the process buffer
        into one training sample.

        Note that the state should be fed to NN on 0..1 scale. Memorywise, it would be
        more efficient to store the arrays as uint8 (0..255) and preprocess
        only the minibatch/single sample at model input.

        https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_atari.py
        """
        imsize = self.img_size
        gray_buffer = [self.preprocess(x) for x in self.process_buffer]
        gray_buffer = [np.reshape(x, x.shape + (1, )) for x in gray_buffer]

        assert len(gray_buffer) == self.num_kept_frames
        return(np.concatenate(gray_buffer, axis=2).astype(np.uint8))

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

        # _, _, error=self.agent.predict_sample(  state, action, reward, next_state,
                                                # done, self.discount)
        # Set error to high value, avoid fitting predicting
        # error = 1. # error[0]
        error  = self.agent.memory.max_error # assign error to the highest available in buffer
        self.agent.memory.add(error, (state, action, reward, next_state, done))

    def random_episode_init(self, n_steps = 30):
        """ Randomize starting position
        """
        # take number from range 0..n_steps as n_steps
        self.process_buffer = []
        n_steps = np.random.randint(self.num_kept_frames, n_steps)
        for i in range(n_steps):
            action = self.env.action_space.sample()
            state,_,_,_ = self.env.step(action)
            if i >= (n_steps - self.num_kept_frames):
                self.process_buffer.append(state)
        return i

    def _init_callbacks(self, num_frames, is_training = True):
        """
        """
        callbacks = []
        callbacks += [UpdateLossMask(clipped_masked_error)]

        log_interval = min(1000, self.num_episodes)
        callbacks += [FileLogger("logs/atari_{}_log.json".format(self.env.spec.id),
                                log_interval)]

        save_interval = min(50000 ,int(self.num_episodes * 1000))
        callbacks += [ModelIntervalCheckpoint("saved_models/dqn_" + \
                                            self.env.spec.id + "_weights_{step}.h5f",
                                            save_interval, verbose = True)]

        callbacks += [History()]
        if is_training:
            # interval = max(1000, round(num_frames//1000))
            callbacks += [TrainEpisodeLogger()]
        else:
            callbacks += [TestLogger()]

        # Define TB callback and  replace epoch with episode
        tboard = SubTensorBoard("./tb_logs", write_graph = False,
                            histogram_freq = 1, write_grads = True)
        ## this is handled in callbacks
        # tboard.on_episode_end = Callback.on_epoch_end # pass
        # tboard.on_epoch_end = KerasCallback.on_epoch_end
        callbacks += [tboard]

        callbacks = CallbackList(callbacks)

        if hasattr(callbacks, 'set_model'):
            callbacks.set_model(self.agent.model)
        else:
            callbacks._set_model(self.agent.model)
        callbacks._set_env(self.env)
        if is_training:
            params = {
                'nb_steps': num_frames,
            }
        else:
            params = {
                'nb_episodes': num_frames,
            }
        if hasattr(callbacks, 'set_params'):
            callbacks.set_params(params)
        else:
            callbacks._set_params(params)

        callbacks.on_train_begin() # Call begining of training/testing callbacks
        return callbacks

    def train(self):
        """ Train the agent to drive up the total reward per episode

        Loop over `num_epsiodes`, each with `max_ep_steps` upmost. Select the
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
        frame_no = np.int16(0)                # The smallest unit of training
        state_no = np.int16(0)                # Each state has `num_kept_frames` frames
        in_memory = np.int16(0)
        was_trained = False
        do_punish = False # Stays so all the time

        logger.info("Initializing memory for {} steps".format(self.min_steps_train))
        while not was_trained:
            # Starting a clean game.
            start_state = state_no
            state_ref = self.env.reset() # Reset the environment, get reference state
            # Randomize starting position
            _ = self.random_episode_init()
            # Obtain random starting state
            state = self.convert_process_buffer()
            done = False

            for st in range(self.max_ep_steps): # self.env._max_episode_steps = self.max_ep_steps
                self.process_buffer = [] # clean up buffer
                # Pick optimal action givent current weights in the predictiove model
                action, q_value = self.agent.policy(state, self.epsilon)

                reward = np.float32(0)      # Reward for `rep_frames` repeats of action
                # DETERMINISTIC FRAME SKIPPING: Fill the buffer
                rep_frames = self.num_kept_frames
                for _ in range(rep_frames): # repeat action over `rep_frames`
                    # callbacks.on_action_begin(action)
                    # TODO: verify that you are not skipping 16 instead of 4
                    state_frame, reward_frame, done_frame, env_info = self.env.step(action)
                    done = done | done_frame        # Are we done? (0 lives)
                    self.process_buffer.append(state_frame)
                    frame_no +=1
                    reward += reward_frame
                    # callbacks.on_action_end(action)
                    if done:
                        break           # dont spill over to the next episode

                for _ in range(self.num_kept_frames - len(self.process_buffer)):
                    # state_frame,_,_,_ = self.env.step(action)
                    self.process_buffer.append(state_frame)
                    # Pad the array with empty observations, if necessary
                    # self.process_buffer.append(np.zeros(state_ref.shape, np.uint8))

                next_state = self.convert_process_buffer() # Preprocess content of buffer
                # Store (s,a,r,s',d) in memory
                self.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state

                state_no = frame_no // self.num_kept_frames
                # PRIORITIZED EXPERIENCE REPLAY: Train with given freq
                if ((state_no % self.model_train_freq == 0) or done) and not was_trained:
                    if isinstance(self.agent.memory, (Memory, )):     # Prioritized
                        in_memory = self.agent.memory.tree.n_entries  # SumTree
                    else:
                        in_memory = len(self.agent.memory)            # deque
                    if in_memory >= self.min_steps_train:
                        # TODO: how to pass Logger instance between modules?
                        logger.info("Model trained, step: {}".format(state_no))
                        _, _ = self.agent.replay(self.discount, logger)
                        was_trained = True

                if done: # each episode is allowed to take max_ep_steps or until we die
                    break

            # # EXPLORATION/EXPLOTATION POLICY: Increase model certainity
            # if (self.epsilon > self.eps_min): #& in_memory >= (self.min_steps_train):
            #     self.epsilon *= self.eps_decay_rate # exponential
            #     # self.epsilon = self.get_curr_eps(ep) # linear

            # Adjust epsilon to account for the init phase
            # Here restart it
            # if was_trained: self.epsilon = self.eps_max

        self.episode_rewards = []   # Store reward achieved at each epsiode
        self.episode_losses = []       # Store mean loss for each episode
        self.episode_durations = []
        # frame_no = np.int16(0)                # The smallest unit of training
        # state_no = np.int16(0)                # Each state has `num_kept_frames` frames
        # in_memory = np.int16(0)
        ale_lives = self.env.unwrapped.ale.lives() # Max number of lives
        callbacks = self._init_callbacks(self.max_frames) # initialize callbacks
        self.is_init = was_trained # dont call callbacks before trained

        for ep in range(self.num_episodes):

            # Starting a clean game.
            start_state = state_no
            state_ref = self.env.reset() # Reset the environment, get reference state

            # Randomize starting position
            _ = self.random_episode_init()

            if self.is_init: callbacks.on_episode_begin(ep)
            # Obtain random starting state
            state = self.convert_process_buffer()

            episode_reward = np.float32(0)
            episode_loss = np.float32(0)
            done = False                 # i.e. we were killed and the game has ended
            num_lives = ale_lives        # How many lives do we start with?

            # Make scalars for TensorBoard. They can be pulled from callbacks as well
            q_vals_ep = []
            sum_action_ep = np.float32(0)

            for st in range(self.max_ep_steps): # self.env._max_episode_steps = self.max_ep_steps
                if self.is_init: callbacks.on_step_begin(st)
                self.process_buffer = [] # clean up buffer
                # Pick optimal action givent current weights in the predictiove model
                action, q_value = self.agent.policy(state, self.epsilon)
                q_vals_ep.append(q_value)
                sum_action_ep += action

                reward = np.float32(0)      # Reward for `rep_frames` repeats of action
                was_killed = False  # Was the agent killed in in this `rep_frames` unit?

                # DETERMINISTIC FRAME SKIPPING: Fill the buffer
                # "We use k = 4 for all games except Space Invaders where we noticed
                # that using k = 4 makes the lasers invisible because of the period
                # at which they blink."
                # May want to have stochastic frame skip
                # rep_frames = random.choice(range(2, self.num_kept_frames + 1))
                # import pdb; pdb.set_trace()
                rep_frames = self.num_kept_frames
                for _ in range(rep_frames): # repeat action over `rep_frames`
                    # callbacks.on_action_begin(action)
                    # TODO: verify that you are not skipping 16 instead of 4
                    state_frame, reward_frame, done_frame, env_info = self.env.step(action)
                    # In original paper, the reward was clipped to work with different games
                    # It is a strong limitation (no way to learn high-reward actions).
                    # Hence we drop it.
                    # reward_frame = np.clip(reward_frame, -1., 1.)
                    done = done | done_frame        # Are we done? (0 lives)
                    self.process_buffer.append(state_frame)
                    frame_no +=1
                    reward += reward_frame
                    # callbacks.on_action_end(action)
                    if done:
                        break           # dont spill over to the next episode

                for _ in range(self.num_kept_frames - len(self.process_buffer)):
                    # Padd array by repeating last seen frame]
                    self.process_buffer.append(state_frame)
                    # Or: Pad the array with empty observations, if necessary
                    # self.process_buffer.append(np.zeros(state_ref.shape, np.uint8))

                if "ale.lives" in env_info and do_punish:
                    if env_info["ale.lives"] != num_lives:
                        num_lives -= 1
                        was_killed = True
                        # Penalize terminal states. Penalize also static actions (??)
                        # e.g. by action==0 or reward==0
                        reward += (was_killed * -10.0) #+ (reward == 0) * -1.0

                episode_reward += reward
                next_state = self.convert_process_buffer() # Preprocess content of buffer
                # Store (s,a,r,s',d) in memory
                self.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state

                state_no = frame_no // self.num_kept_frames

                # PRIORITIZED EXPERIENCE REPLAY: Train with given freq
                if (state_no % self.model_train_freq == 0) or done:
                    if isinstance(self.agent.memory, (Memory, )):     # Prioritized
                        in_memory = self.agent.memory.tree.n_entries  # SumTree
                    else:
                        in_memory = len(self.agent.memory)            # deque
                    if in_memory >= self.min_steps_train: ## always true once init
                        # TODO: how to pass Logger instance between modules?
                        logger.debug("Model trained, step: {}".format(state_no))
                        metrics, val_data = self.agent.replay(self.discount, logger)
                        episode_loss += metrics[0]
                        was_trained = True
                        callbacks.callbacks[-1].validation_data = val_data

                # TARGET DQN: Update target NN less frequently than model DQN
                if (state_no%self.target_update_freq == (self.target_update_freq-1)):
                    self.agent.target_update()
                    logger.info("Target updated, step: {}".format(state_no))

                if self.verbose: # Report some values for debugging
                    if (state_no%self.qa_report_freq == (self.qa_report_freq - 1)):
                        logger.debug("Action: {}, Q_val: {},  eps: {}".\
                                    format(action, q_value, self.epsilon))

                if self.is_init:
                    callbacks.on_step_end(ep, { 'action':action,
                                                'observation':state,
                                                'reward':reward,
                                                'metrics': metrics,
                                                'episode': ep,
                                                'info':env_info,
                                                'epsilon': self.epsilon,
                                                })
                if done: # each episode is allowed to take max_ep_steps or until we die
                    break

            # EXPLORATION/EXPLOTATION POLICY: Increase model certainity
            if (self.epsilon > self.eps_min): #& in_memory >= (self.min_steps_train):
                # self.epsilon *= self.eps_decay_rate # exponential
                self.epsilon = self.get_curr_eps(ep) # linear

            self.episode_rewards.append(episode_reward)
            self.episode_durations.append(state_no - start_state)
            self.episode_losses.append( episode_loss / (state_no - start_state) * \
                                        self.model_train_freq)

            if self.is_init:
                # Values will be passed to tensorboard scalar summary
                nsteps_ep = state_no - start_state
                ep_log = {  'episode_reward': episode_reward,
                            'nb_episode_steps': nsteps_ep,
                            'nb_steps': state_no,
                            'mean_q_val': np.mean(q_vals_ep),
                            'min_q_val': np.min(q_vals_ep),
                            'max_q_val': np.max(q_vals_ep),
                            'mean_action': sum_action_ep / nsteps_ep,
                            'mean_reward' : episode_reward /(nsteps_ep * self.num_kept_frames),
                            'mean_loss' : episode_loss / nsteps_ep * self.model_train_freq,
                            'epsilon': np.float32(self.epsilon)
                            }
                callbacks.on_episode_end(ep, ep_log)

            if self.verbose: # Report some values for debugging
                if (((ep % 15) == 0) and ep > 0):
                    mean_reward = np.mean(self.episode_rewards[max(ep-15, 0):ep+1])
                    logger.info("Episode {}/{} finished. Mean reward over last 15 episodes: {:.2f}"\
                        .format(ep, self.num_episodes, mean_reward))
                else:
                    logger.debug("Episode {}/{} finished. Reward: {:.2f}, Frame # {}."\
                        .format(ep+1, self.num_episodes, episode_reward, frame_no))


            if ((ep % self.model_save_freq) == 0 and ep > 0):
                logger.debug("Saving Network, frame: {}, ep: {}".format(frame_no, ep+1))
                self.agent.save_network("saved_models/" + self.env.spec.id + \
                                        "_" + self.mode+"_fr"+str(frame_no)+".h5")

            if frame_no >= self.max_frames or ep == (self.num_episodes - 1) :
                logger.info("Frame # {}. End of Training.".format(frame_no))
                logger.info("Saving Network, frame: {}, ep: {}".format(frame_no, ep+1))
                self.agent.save_network("saved_models/" + self.env.spec.id + \
                                        "_" + self.mode+"_ep"+str(ep+1)+".h5")
                break

        callbacks.on_train_end(logs={})

    def plot_(self):
        """Plot all epsiode rewards,losses,lengths collected throughout training
        """
        try:
            plt.clf()
            fig, ax  = plt.subplots(1, 1)
            ax.plot(self.episode_rewards)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            fig.savefig("summary/" + self.env.spec.id + "_Erewards.png")
            logger.info("Plot of Rewards was saved.")

        except NameError as e: # or whatever gets thrown if not defined
            raise e

        try:
            plt.clf()
            fig, ax  = plt.subplots(1, 1)
            ax.plot(self.episode_losses)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Mean Replay Loss")
            fig.savefig("summary/" + self.env.spec.id + "_Elosses.png")
            logger.info("Plot of Losses was saved.")

        except NameError as e: # or whatever gets thrown if not defined
            raise e

        try:
            plt.clf()
            fig, ax  = plt.subplots(1, 1)
            ax.plot(self.episode_durations)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Number of States")
            fig.savefig("summary/" + self.env.spec.id + "_Elengths.png")
            logger.info("Plot of Episode Lengths was saved.")

        except NameError as e: # or whatever gets thrown if not defined
            raise e

    def simulate_(self):
        """Simple game visualizer

        Usses ffmpeg to convert simulated images into animated gif. Make sure
        it is installed on your machine.
        """
        # Clean up anim direcotory contents
        folder = os.path.abspath("anim")
        for f in os.listdir(folder):
            fpath = os.path.join(folder, f)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

        done = False
        tot_award = 0
        state = self.env.reset()
        frame_no = 0
        while not done:
            frame_no += 1
            state = self.convert_process_buffer()
            # Take deterministic action (eps=0) according to model.
            action = self.agent.policy(state, epsilon = 0.05)[0]
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

    def test(self, num_episodes = 10):
        self.episode_rewards = []   # Store reward achieved at each epsiode
        self.episode_losses = []       # Store mean loss for each episode
        self.episode_durations = []
        frame_no = 0                # The smallest unit of training
        state_no = 0                # Each state has `num_kept_frames` frames
        initial_state = self.convert_process_buffer()
        ale_lives = self.env.unwrapped.ale.lives() # Max number of lives
        callbacks = self._init_callbacks(self.num_episodes, is_training = False)

        self.epsilon = 0
        self.num_episodes = num_episodes

        for ep in range(self.num_episodes):
            # Starting a clean game.
            start_state = state_no
            state_ref = self.env.reset() # Reset the environment, get reference state

            # Randomize starting position
            _ = self.random_episode_init()

            callbacks.on_episode_begin(ep)

            # Do we always start from the same spot?
            # If so, neednt to reinitialize buffer
            # self.process_buffer = self._init_process_buffer()#`num_kept_frames` reps of nonaction
            # initial_state = self.convert_process_buffer() # Preprocess content of buffer
            state = initial_state        # initial state == current state at each ep start

            episode_reward = 0           # This one should be increasing with more training
            episode_loss = 0
            done = False                 # i.e. we were killed and the game has ended
            num_lives = ale_lives        # How many lives do we start with?

            for st in range(self.max_ep_steps): # self.env._max_episode_steps = self.max_ep_steps
                callbacks.on_step_begin(st)
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
                    # callbacks.on_action_begin(action)
                    # TODO: verify that you are not skipping 16 instead of 4
                    state_frame, reward_frame, done_frame, env_info = self.env.step(action)
                    # In original paper, the reward was clipped to work with different games
                    # It is a strong limitation (no way to learn high-reward actions).
                    # Hence we drop it.
                    # reward_frame = np.clip(reward_frame, -1., 1.)
                    done = done | done_frame        # Are we done? (0 lives)
                    self.process_buffer.append(state_frame)
                    frame_no +=1
                    reward += reward_frame
                    # callbacks.on_action_end(action)
                    if done:
                        break           # dont spill over to the next episode

                for _ in range(self.num_kept_frames - len(self.process_buffer)):
                    # Pad the array with empty observations, if necessary
                    self.process_buffer.append(np.zeros(state_ref.shape, np.uint8))

                do_punish = False
                if "ale.lives" in env_info and do_punish:
                    if env_info["ale.lives"] != num_lives:
                        num_lives -= 1
                        was_killed = True
                        # Penalize terminal states. Penalize also static actions (??)
                        # e.g. by action==0 or reward==0
                        reward += (was_killed * -10.0) #+ (reward == 0) * -1.0

                episode_reward += reward
                next_state = self.convert_process_buffer() # Preprocess content of buffer
                # make next_state the new current state for the next frame.
                state = next_state

                state_no = frame_no // self.num_kept_frames

                if self.verbose: # Report some values for debugging
                    if (state_no%self.qa_report_freq == (self.qa_report_freq - 1)):
                        logger.debug("Action: {}, Q_val: {},  eps: {}".\
                                    format(action, q_value, self.epsilon))

                callbacks.on_step_end(ep, { 'action':action,
                                            'state':state,
                                            'reward':reward,
                                            'episode': ep,
                                            'info':env_info})
                if done: # each episode is allowed to take max_ep_steps or until we die
                    break


            self.episode_rewards.append(episode_reward)
            self.episode_durations.append(state_no - start_state)
            self.episode_losses.append( episode_loss / (state_no - start_state) * \
                                        self.model_train_freq)

            callbacks.on_episode_end(ep,{'episode_reward': episode_reward,
                                        'nb_episode_steps': state_no - start_state,
                                        'nb_steps': state_no})
                                        # Should plot mean Q, Actions Distribution
                                        # Increase weigth regularizer? -- probably not

            if self.verbose: # Report some values for debugging
                if (((ep % 15) == 0) and ep > 0):
                    mean_reward = np.mean(self.episode_rewards[max(ep-15, 0):ep+1])
                    logger.debug("Episode {}/{} finished. Mean reward over last 15 episodes: {:.2f}"\
                        .format(ep, self.num_episodes, mean_reward))
                else:
                    logger.debug("Episode {}/{} finished. Reward: {:.2f}, Frame # {}."\
                        .format(ep+1, self.num_episodes, episode_reward, frame_no))

            if frame_no >= self.max_frames or ep == (self.num_episodes - 1) :
                logger.debug("Frame # {}. End of Test.".format(frame_no))
                break

        callbacks.on_train_end(logs={})

    def main(self):
        """ Training wrapper
        """
        logger = initialize_logger()
        try:
            self.train()
        finally:
            num_ep = len(self.episode_rewards)
            self.agent.save_network("saved_models/{}_ep{}_final.h5".format(self.env.spec.id, num_ep))
            teardown_logger(logger)

## ----------------------------------------------------------------------------
## Execution
NUM_EPISODES = 3000
MIN_STEPS_TRAIN = 30000
EPSILON = .5
LEARN_RATE = 0.01 # previously 0.00025 which seems very low
LOAD_PATH = "./saved_models/BreakoutDeterministic-v4_ep1500_final.h5"
# to check: decay, l2, loss (could be uninformative?)
if __name__ == '__main__':
    game = AtariGame(num_episodes = NUM_EPISODES, min_steps_train = MIN_STEPS_TRAIN,
                    epsilon = EPSILON, learn_rate = LEARN_RATE, load_path = LOAD_PATH)
    game.main()
