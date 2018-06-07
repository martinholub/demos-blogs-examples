import numpy as np
import random

import keras
from keras.models import load_model, Sequential, Model, model_from_config
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam #Adagrad, RMSProp
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.callbacks import Callback, CallbackList

from prioritized_memory import PERMemory as Memory
from utils import clipped_error, mean_q

class DuelQ(object):
    """Constructs the desired deep q learning network"""
    def __init__(self, learn_rate, img_size, num_frames, action_size, replay_size):
        self.img_size = tuple(img_size) # tuple
        self.num_frames = int(num_frames)
        self.learn_rate = float(learn_rate)
        self.action_size = int(action_size)
        self.replay_size = int(replay_size)
        self.num_epochs = int(1)
        self._construct_q_network()


    def _construct_q_network(self):
        # Extends the network architecture found in DeepMind paper

        model = Sequential()

        model.add(Convolution2D(filters = 32, kernel_size = (8, 8), strides = (4, 4),
                                input_shape = self.img_size + (self.num_frames, )))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Convolution2D(filters = 64, kernel_size = (4, 4), strides = (2, 2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Convolution2D(filters = 64, kernel_size = (3, 3), strides = (1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        flatten = model.layers[-1].output # get output of the flatten layer

        fc1 = Dense(units = 512, activation = None)(flatten)
        advantage = Dense(self.action_size, activation = None)(fc1)
        fc2 = Dense(units = 512, activation = None)(flatten)
        value = Dense(1)(fc2)

        # dueling_type == 'avg'
        # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
        policy = Lambda(lambda x: x[0]-K.mean(x[0])+x[1],
                        output_shape = (self.action_size, ))([advantage, value])

        input_layer = model.input
        self.model = Model(inputs = [input_layer], outputs = [policy])
        config = self.model.get_config()
        self.target_model = Model.from_config(config)
        self.target_update()
        del(model)

        losses = [clipped_error]
        metrics = ["mae", mean_q]
        self.model.compile( loss = losses, optimizer = Adam(lr = self.learn_rate),
                            metrics = metrics)
        self.target_model.compile(  loss = 'MSE', optimizer = Adam(lr = self.learn_rate),
                                    metrics = metrics)

        print(self.model.summary())

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


    # def predict_sample(self, state, action, reward, next_state, done, discount):
    #     target = reward
    #     if not done:
    #         # predict the future discounted reward
    #         q_next=self.target_model.predict(\
    #                         next_state.reshape(1, *self.img_size, self.num_frames),
    #                         batch_size = 1)
    #         target = reward + discount * np.amax(q_next) # == q_hat
    #
    #     # make the agent approx. map the current state to future discounted reward
    #     q_target = self.model.predict(state.reshape(1, *self.img_size, self.num_frames),
    #                                 batch_size = 1)
    #     target_old = q_target[0][action] # pull out old value of q_hat
    #     q_target[0][action] = target # update q to future
    #
    #     # Get error for updating priorities in the memory
    #     error = abs(target_old - target)
    #
    #     return (state, q_target, error)

    def predict_batch(self, minibatch, discount):
        #Should work also on batch size 1
        batch_size = len(minibatch)
        # minibatch = np.reshape(minibatch, (batch_size, 5))
        # states, actions, rewards, next_states, dones = np.split(minibatch, 5, axis = 1)

        states = []
        rewards = []
        actions = []
        next_states = []
        dones = []
        for j, (state, action, reward, next_state, done) in enumerate(minibatch):
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        q_olds = self.model.predict_on_batch(np.reshape(states, (-1,*self.img_size,self.num_frames)))
        q_nexts = self.model.predict_on_batch(np.reshape(next_states, (-1,*self.img_size,self.num_frames)))
        next_actions = np.argmax(q_nexts, axis=1)
        q_targets = self.target_model.predict_on_batch(np.reshape(next_states, (-1,*self.img_size,self.num_frames)))
        # predict the future discounted reward. target == reward if done
        targets =   rewards + \
                    discount * np.invert(dones).astype(np.float32) * \
                    q_targets[range(batch_size), next_actions]
        targets_old = [q[a] for q,a in zip(q_olds, actions)] # pull out old value of q_hat
        # make the agent approx. map the current state to future discounted reward
        q_targets[range(batch_size), actions] = targets # update q to future
        # Get error for updating priorities in the memory
        errors = (abs(targets_old - targets))
        # state is x, q_target is y
        states = np.stack(states, axis = 0) # creates new axis
        # q_targets = np.concatenate(q_targets, axis = 0)

        return (states, q_targets, errors)

    def predict_sample(self, state, action, reward, next_state, done, discount):
        dummy_batch = [[state, action, reward, next_state, done]]
        state, q_target, error = self.predict_batch(dummy_batch, discount)
        return state, q_target, error

    def replay(self, memory, discount):
        ## Simple Style Draw random minibatch sample from memory
        # minibatch=random.sample(self.memory, min(len(memory), self.replay_size)
        ## Prioretized memory style
        minibatch, idxs, is_weights = memory.sample(self.replay_size)
        states, q_targets, errors = self.predict_batch(minibatch, discount)
        memory.batch_update(idxs, errors)
        self.model.train_on_batch(states, q_targets, sample_weight = is_weights)

        return memory
        # loss = self.model.train_on_batch(s_batch, targets, sample_weight=is_weights)

        # # update priority
        # if isinstance(self.memory, (Memory, )):
        #     for i in range(len(minibatch)):
        #         idx = idxs[i]
        #         self.memory.update(idx, errors[i])


    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print("Successfully saved network.")

    def load_network(self, path):
        self.model.load_weights(path)
        self.target_model.load_weights(path)
        print("Succesfully loaded network.")

    def target_update(self):
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)

if __name__ == "__main__":
    print("Haven't finished implementing yet...'")
    space_invader = SpaceInvader()
    space_invader.load_network("duel_saved.h5")
    # print space_invader.calculate_mean()
    space_invader.simulate("duel_q_video_2", True)
    # space_invader.train(TOT_FRAME)
