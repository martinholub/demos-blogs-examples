import numpy as np
import random

import keras
from keras.models import load_model, Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam #Adagrad, RMSProp
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from prioritized_memory import PERMemory as Memory


class DeepQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self, learn_rate, img_size, num_frames, action_size, replay_size):
        self.img_size = tuple(img_size) # tuple
        self.num_frames = int(num_frames)
        self.learn_rate = float(learn_rate)
        self.action_size = int(action_size)
        self.replay_size = int(replay_size)
        self.num_epochs = int(1)
        self.tau = float(0.01)
        self._construct_q_network()

    def _construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        self.model.add(Convolution2D(   filters = 32, kernel_size = (8, 8),
                                        strides = (4, 4),
                                        input_shape = self.img_size + (self.num_frames, )))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(filters = 64, kernel_size = (4, 4), strides = (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(filters = 64, kernel_size = 3, strides = 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units = 512, activation = None))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.action_size, activation = None))
        self.model.compile(loss = 'MSE', optimizer = Adam(lr = self.learn_rate))

        # Creates a target network as described in DeepMind paper
        self.target_model = Sequential()
        self.target_model.add(Convolution2D(   filters = 32, kernel_size = (8, 8),
                                        strides = (4, 4),
                                        input_shape = self.img_size + (self.num_frames, )))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(filters = 64, kernel_size = (4, 4), strides = (2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(filters = 64, kernel_size = 3, strides = 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(units = 512, activation = None))
        # self.target_model.add(Activation('relu'))
        self.target_model.add(Dense(self.action_size, activation = None))
        self.target_model.compile(loss = 'MSE', optimizer = Adam(lr = self.learn_rate))
        self.target_model.set_weights(self.model.get_weights())

        print(self.model.summary())
        print(self.target_model.summary())
        print("Successfully constructed networks.")

    def policy(self, state, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""

        q_actions = self.model.predict( state.reshape(1, *self.img_size, self.num_frames),
                                        batch_size = 1)
        if np.random.rand() < epsilon:
            opt_policy = random.choice(range(self.action_size))
        else:
            opt_policy = np.argmax(q_actions)

        return opt_policy, q_actions[0, opt_policy]

    def learn_sample(self, state, action, reward, next_state, done, discount):
        #TODO: call it predict sample
        target = reward
        if not done:
            # predict the future discounted reward
            q_next=self.target_model.predict(\
                            next_state.reshape(1, *self.img_size, self.num_frames),
                            batch_size = 1)
            target = reward + discount * np.amax(q_next) # == q_hat

        # make the agent approx. map the current state to future discounted reward
        q_target = self.model.predict(state.reshape(1, *self.img_size, self.num_frames),
                                    batch_size = 1)
        target_old = q_target[0][action] # pull out old value of q_hat
        q_target[0][action] = target # update q to future

        # Get error for updating priorities in the memory
        error = abs(target_old - target)

        return (state, q_target, error)

    def replay(self, memory, discount):
        ## Prioretized memory style
        minibatch, idxs, is_weights = memory.sample(self.replay_size)
        q_targets = []
        states = []

        # logger.debug("Importance Sampling Weights: {}".format(is_weights))
        # Update model by looping over sampled experiences, one by one (!)
        q_targets = []
        states = []
        for j, (state, action, reward, next_state, done) in enumerate(minibatch):
            _, q_target, error = self.learn_sample( state, action, reward,
                                                    next_state, done, discount)
            memory.update(idxs[j], error)
            q_targets.append(q_target)
            states.append(state)

        # state is x, q_target is y
        states = np.stack(states, axis = 0) # creates new axis
        q_targets = np.concatenate(q_targets, axis = 0)
        # TODO: try fit on batch
        self.model.fit( states, q_targets, epochs = self.num_epochs,
                        batch_size = len(minibatch), verbose = 0,
                        sample_weight = is_weights)
                        # batch_size = self.mp.replay_size

        return memory

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model = load_model(path)
        print("Succesfully loaded network.")

    def target_update(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in range(len(model_weights)):
            target_model_weights[i] =   self.tau * model_weights[i] + \
                                        (1 - self.tau) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

if __name__ == "__main__":
    print("Haven't finished implementing yet...'")
    space_invader = SpaceInvader()
    space_invader.load_network("saved.h5")
    # print space_invader.calculate_mean()
    space_invader.simulate("deep_q_video", True)
    # space_invader.train(TOT_FRAME)
