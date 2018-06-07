#from keras import backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import gym
import gym.spaces
gym.logger.set_level(40) # Disable printing of warn level log to stdout

import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from prioritized_memory import Memory

from logger_utils import initialize_logger, teardown_logger
logger = initialize_logger()

class envParams(object):
    """Parameters of the environment
    """
    def __init__(self, env_id = "CartPole-v0", action_size = 2, state_size = 4,
                discount = 1.0, epsilon = 0.2, eps_decay_rate = 0.995,
                eps_min = 0.001, num_episodes = 50, max_steps = 1000,
                reward_solve = 195, n_epochs_solve = 100):
        self.env_id = str(env_id) # which enronment are we solving
        self.action_size = int(action_size) # actions dimension
        self.state_size = int(state_size) # observations dimension
        self.discount = float(discount) # discount rate (also gamma)
        self.epsilon = float(epsilon) # exploration rate
         # epsilon decay rate ~ model certainity
        self.eps_decay_rate = float(eps_decay_rate)
        self.eps_min = float(eps_min) # minimal exploration rate
        self.num_episodes = num_episodes # number of trials of the game
        self.max_steps = int(max_steps) # Upper lomit on ticks in each episode
        # Value of reward that solves the game at each epoch
        self.reward_solve = int(reward_solve)
         # Over how many epochs we need  to keep the reward?
        self.n_epochs_solve = int(n_epochs_solve)

class modelParams(object):
    """Parameters of the model
    """
    def __init__(self, learn_rate = 0.001, replay_size = 32, max_memory = 2000,
                num_epochs = 2, hidden_size = 16):
        self.learn_rate = float(learn_rate) # optimizer learning rate
        # batch size for update of agent after epsiode
        self.replay_size = int(replay_size)
        self.max_memory = int(max_memory) # size of memory for past model inputs
        self.num_epochs = int(num_epochs) # number of epochs for model.fit()
        self.hidden_size = int(hidden_size) # hidden layer size for NN

class dqnAgent(object):
    """Deep Q-Learning Agent
    """
    def __init__(self, model_params, env_params):
        self.ep = env_params # Environemtn Parameters
        self.mp = model_params # Model Parameters
        self.model = self._build_model()
        # deque: list-like, optimzied for fast access at either end
        # self.memory = deque(maxlen = int(model_params.max_memory))
        # Prioritized Memory class implementing Sum Tree
        self.memory = Memory(model_params.max_memory)

    def _build_model(self):
        """Build a NN model for Q-learning task
        """
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # Layer 1
        hs = self.mp.hidden_size
        model.add(Dense(hs, input_shape = (self.ep.state_size, )))
        #model.add(BatchNormalization())
        model.add(Activation(activation = 'relu'))
        #model.add(Dropout(0.5))

        # Layer 2
        model.add(Dense(hs)) #
        #model.add(BatchNormalization())
        model.add(Activation(activation = 'relu'))
        #model.add(Dropout(0.5))

        # Output
        model.add(Dense(self.ep.action_size, activation = 'linear'))

        model.compile(  loss = "MSE",
                        optimizer = Adam(lr=self.mp.learn_rate, amsgrad = True)
                        )
        logger.info(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        """Prevent NN forgeting about past experiences
        """
        self.memory.append((state, action, reward, next_state, done))

    def remember_priority(self, state, action, reward, next_state, done):
        """Remeber also error at current single step

        This allows us to sample from memory with priority.
        Of course, it also slows us down a bit.
        """
        target = reward # == 1
        if not done:
            # predict the future discounted reward
            q_next = self.model.predict(next_state)
            target = reward + self.ep.discount * np.amax(q_next) # == q_hat

        # make the agent approx. map the current state to future discounted reward
        q_target = self.model.predict(state) # old value of q
        target_old = q_target[0][action] # pull out old value of q_hat
        q_target[0][action] = target # update q to future

        # Update priorities in memory
        error = abs(target_old - target)
        self.memory.add(error, (state, action, reward, next_state, done))

    def policy(self, state):
        """Decision making based on current observation (state)
        """
        # Occasionaly, randomly explore
        if np.random.rand() < self.ep.epsilon:
            return random.choice(range(self.ep.action_size))
        else:
            # Let model decide what to do next
            act_values = self.model.predict(state)
            return np.argmax(act_values)  # returns action

    def replay(self):
        """Train model on samples drawn from its memory

        Done after each epoch. replay_size shows to be important parameter and should
        be big enough. Idea of replay is inspired by neuroscience and how people learn
        from past experiences during rest periods (sleep).

        References:
          https://www.quora.com/Artificial-Intelligence-What-is-an-intuitive-explanation-of-how-deep-Q-networks-DQN-work
          https://arxiv.org/pdf/1511.05952.pdf
        """
        # Draw random minibatch sample from memory
        ## Simple Style
        # minibatch=random.sample(self.memory,
        #                         min(len(self.memory), self.mp.replay_size))

        ## Prioretized memory style
        minibatch, idxs, is_weights = self.memory.sample(self.mp.replay_size)
        q_targets = []
        states = []

        logger.debug("Importance Sampling Weights: {}".format(is_weights))
        # Update model by looping over sampled experiences, one by one (!)
        for j, (state, action, reward, next_state, done) in enumerate(minibatch):

            target = reward # == 1
            if not done:
                # predict the future discounted reward
                q_next = self.model.predict(next_state)
                target = reward + self.ep.discount * np.amax(q_next) # == q_hat

            # make the agent approx. map the current state to future discounted reward
            q_target = self.model.predict(state) # old value of q
            target_old = q_target[0][action] # pull out old value of q_hat
            q_target[0][action] = target # update q to future

            # Update priorities in memory
            error = abs(target_old - target)
            self.memory.update(idxs[j], error)

            # state is x, q_target is y
            self.model.fit( state, q_target, epochs = self.mp.num_epochs,
                            batch_size = 1, verbose = 0, sample_weight=[is_weights[[j]]])
                        # batch_size = self.mp.replay_size

        # # update priority
        # if isinstance(self.memory, (Memory, )):
        #     for i in range(len(minibatch)):
        #         idx = idxs[i]
        #         self.memory.update(idx, errors[i])


class cartPoleSolver(object):
    """CartPole-v0 solver
    """
    def __init__(self, agent, verbose = False):
        # unpack agent class
        self.agent = agent
        self.mp = agent.mp
        self.ep = agent.ep
        self.verbose = verbose

    def train(self, do_plot = False):
        env = gym.make(self.ep.env_id)
        # Empirically, long training periods produce more skilled agents
        # Default max is 200, which is also condition for CartPole
        env._max_episode_steps = self.ep.max_steps

        episode_rewards = []
        frames = []
        for ep in range(self.ep.num_episodes):
            # Reset state at the beginning of each game
            state = env.reset()
            state = np.reshape(state, [1, self.ep.state_size])
            rewards = []
            for t in range(self.ep.max_steps):
                # Take smart action based on defined policy
                action = self.agent.policy(state)
                # Advance the game to the next frame based on the action.
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, self.ep.state_size])

                rewards.append(reward)
                # Make agent remeber past experiences
                # self.agent.remember(state, action, reward, next_state, done)
                self.agent.remember_priority(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame.
                state = next_state

                if do_plot:
                    # Render into buffer (slows down things quite a bit)
                    frame = env.render(mode = 'rgb_array')
                    frames.append(frame)

                if done: # Finish if enviroment solved
                    break

            # Increase model certainity
            if self.ep.epsilon > self.ep.eps_min:
                self.ep.epsilon *= self.ep.eps_decay_rate

            # train the agent with the experience of preceding episodes
            # Here the training frequency is once per epoch only
            # if self.agent.memory.tree.n_entries >= self.mp.replay_size:
            self.agent.replay()

            episode_rewards.append(np.sum(rewards))
            mean_reward=np.mean(episode_rewards[max(ep-self.ep.n_epochs_solve, 0):ep+1])

            # Report on progress - did we solve the task already?
            if mean_reward >= self.ep.reward_solve and ep >= self.ep.n_epochs_solve:
                print("Episodes before solve {}".format(ep-100+1))
                break
            if (self.verbose and ((ep % 100) == 0) and ep > 0):
                print("Episode {}/{} finished. Mean reward over last 100 episodes: {:.2f}"\
                      .format(ep, self.ep.num_episodes, mean_reward))

        env.close()
        self.episode_rewards = episode_rewards
        self.frames = frames

    def plot_(self):
        """Simple plot wrapper
        """
        try:
            plt.clf()
            fig, ax  = plt.subplots(1, 1)
            ax.plot(self.episode_rewards)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.axhline(self.ep.reward_solve, 0, 1, alpha = 0.5, ls = "--", lw = "1.0",
                    color = "firebrick")
            plt.show()
        except NameError as e: # or whatever gets thrown if not defined
            raise e

    def animate_(self, name = "cartpole", do_save = True):
        """Simple game visualizer
        """
        plt.clf()
        fig, ax = plt.subplots(1, 1)

        patch = plt.imshow(self.frames[0])
        plt.axis('off')

        if do_save:
            for i in range(len(self.frames)):
                patch.set_data(self.frames[i])
                plt.savefig("pics/" + name + "_" + str(i) + ".png")
        else:
            def animate_helper(i):
                patch.set_data(self.frames[i])
            anim = animation.FuncAnimation( fig, animate_helper,
                                            frames=len(frames),
                                            interval=5, blit = False, repeat=False)

    def main(self):
        """Game wrapper
        """
        logger = initialize_logger()
        logger.info("Starting `cartpole` game.")
        logger.info("Environment parameters: \n{}".format(self.ep.__dict__))
        logger.info("Model parameters: \n{}".format(self.mp.__dict__))
        do_anim = False
        try:
            self.train(do_anim)
            self.plot_()
            if do_anim:
                self.animate_()
            logger.info("Game finished.")
        finally:
            teardown_logger(logger)
